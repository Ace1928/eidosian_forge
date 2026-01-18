from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
def test_frompyfunc():
    myadd = da.frompyfunc(add, 2, 1)
    np_myadd = np.frompyfunc(add, 2, 1)
    x = np.random.normal(0, 10, size=(10, 10))
    dx = da.from_array(x, chunks=(3, 4))
    y = np.random.normal(0, 10, size=10)
    dy = da.from_array(y, chunks=2)
    assert_eq(myadd(dx, dy), np_myadd(x, y))
    assert_eq(myadd.outer(dx, dy), np_myadd.outer(x, y))
    with pytest.raises(NotImplementedError):
        da.frompyfunc(lambda x, y: (x + y, x - y), 2, 2)