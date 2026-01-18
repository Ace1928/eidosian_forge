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
@pytest.mark.parametrize('ufunc', binary_ufuncs)
def test_binary_ufunc(ufunc):
    dafunc = getattr(da, ufunc)
    npfunc = getattr(np, ufunc)
    arr1 = np.random.randint(1, 100, size=(20, 20))
    darr1 = da.from_array(arr1, 3)
    arr2 = np.random.randint(1, 100, size=(20, 20))
    darr2 = da.from_array(arr2, 3)
    assert isinstance(dafunc(darr1, darr2), da.Array)
    assert_eq(dafunc(darr1, darr2), npfunc(arr1, arr2))
    assert isinstance(npfunc(darr1, darr2), da.Array)
    assert_eq(npfunc(darr1, darr2), npfunc(arr1, arr2))
    assert isinstance(dafunc(arr1, arr2), np.ndarray)
    assert_eq(dafunc(arr1, arr2), npfunc(arr1, arr2))
    assert isinstance(dafunc(darr1, 10), da.Array)
    assert_eq(dafunc(darr1, 10), npfunc(arr1, 10))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert isinstance(dafunc(10, darr1), da.Array)
        assert_eq(dafunc(10, darr1), npfunc(10, arr1))
    assert isinstance(dafunc(arr1, 10), np.ndarray)
    assert_eq(dafunc(arr1, 10), npfunc(arr1, 10))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert isinstance(dafunc(10, arr1), np.ndarray)
        assert_eq(dafunc(10, arr1), npfunc(10, arr1))