from __future__ import annotations
from operator import add
from time import sleep
import pytest
from dask.cache import Cache
from dask.callbacks import Callback
from dask.local import get_sync
from dask.threaded import get
def test_cache_with_number():
    c = Cache(10000, limit=1)
    assert isinstance(c.cache, cachey.Cache)
    assert c.cache.available_bytes == 10000
    assert c.cache.limit == 1