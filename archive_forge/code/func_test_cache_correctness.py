from __future__ import annotations
from operator import add
from time import sleep
import pytest
from dask.cache import Cache
from dask.callbacks import Callback
from dask.local import get_sync
from dask.threaded import get
def test_cache_correctness():
    c = Cache(10000)
    da = pytest.importorskip('dask.array')
    from numpy import ones, zeros
    z = da.from_array(zeros(1), chunks=10)
    o = da.from_array(ones(1), chunks=10)
    with c:
        assert (z.compute() == 0).all()
        assert (o.compute() == 1).all()