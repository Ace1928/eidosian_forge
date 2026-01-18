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
@pytest.mark.parametrize('func', ['i0', 'sinc', 'nan_to_num'])
def test_non_ufunc_others(func):
    arr = np.random.randint(1, 100, size=(20, 20))
    darr = da.from_array(arr, 3)
    dafunc = getattr(da, func)
    npfunc = getattr(np, func)
    assert_eq(dafunc(darr), npfunc(arr), equal_nan=True)