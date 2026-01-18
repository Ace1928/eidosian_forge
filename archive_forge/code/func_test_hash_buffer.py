from __future__ import annotations
import pytest
from dask.hashing import hash_buffer, hash_buffer_hex, hashers
@pytest.mark.parametrize('x', buffers)
def test_hash_buffer(x):
    for hasher in [None] + hashers:
        h = hash_buffer(x, hasher=hasher)
        assert isinstance(h, bytes)
        assert 8 <= len(h) < 32
        assert h == hash_buffer(x, hasher=hasher)