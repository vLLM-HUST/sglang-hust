from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.common import release_kv_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Allocator:
    def __init__(self):
        self.freed = []
        self.page_size = 1

    def free_segments(self, segments):
        self.freed.append([(indices.clone(), start) for indices, start in segments])


class _ReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.tensor(
            [[100, 101, 102, 200, 201, 202, 300, 301]], dtype=torch.int32
        )
        self.freed = []

    def free(self, req):
        self.freed.append(req)


class _TreeCache:
    def __init__(self):
        self.req_to_token_pool = _ReqToTokenPool()
        self.token_to_kv_pool_allocator = _Allocator()
        self.dec_locked = []
        self.released_refs = []

    def supports_mamba(self):
        return False

    def cache_finished_req(self, req, is_insert=True):
        raise AssertionError("stitched release must bypass native prefix insertion")

    def dec_lock_ref(self, node):
        self.dec_locked.append(node)

    def segment_reuse_release_req_body_refs(self, req):
        self.released_refs.append(req.segment_reuse_borrowed_body_object_id)
        req.segment_reuse_borrowed_body_object_id = None


def _request(body_indices=(200, 201, 202)):
    req = SimpleNamespace(
        req_pool_idx=0,
        mamba_pool_idx=None,
        last_node="node-1",
        kv_committed_len=8,
        kv=SimpleNamespace(kv_allocated_len=8),
        segment_reuse_body_insert_pos=3,
        segment_reuse_body_kv_indices=torch.tensor(body_indices, dtype=torch.int64),
        segment_reuse_borrowed_body_object_id="body-1",
    )
    req.uses_segment_reuse_stitch = lambda: True

    def effective_kv_committed_len():
        return req.kv_committed_len

    req.effective_kv_committed_len = effective_kv_committed_len
    return req


def test_stitched_release_frees_only_request_owned_indices():
    cache = _TreeCache()
    req = _request()

    release_kv_cache(req, cache)

    assert [
        [(indices.tolist(), start) for indices, start in call]
        for call in cache.token_to_kv_pool_allocator.freed
    ] == [
        [([100, 101, 102], 0), ([300, 301], 6)]
    ]
    assert cache.dec_locked == ["node-1"]
    assert cache.released_refs == ["body-1"]
    assert cache.req_to_token_pool.freed == [req]
    assert req.segment_reuse_borrowed_body_object_id is None
    assert req.kv is None


def test_stitched_release_identity_mismatch_fails_before_mutation():
    cache = _TreeCache()
    req = _request(body_indices=(200, 999, 202))

    with pytest.raises(RuntimeError, match="identity"):
        release_kv_cache(req, cache)

    assert cache.token_to_kv_pool_allocator.freed == []
    assert cache.dec_locked == []
    assert cache.released_refs == []
    assert cache.req_to_token_pool.freed == []
    assert req.kv is not None
