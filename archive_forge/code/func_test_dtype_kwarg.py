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
@pytest.mark.parametrize('dt', ['float64', 'float32', 'int32', 'int64'])
def test_dtype_kwarg(dt):
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    darr1 = da.from_array(arr1)
    darr2 = da.from_array(arr2)
    expected = np.add(arr1, arr2, dtype=dt)
    result = np.add(darr1, darr2, dtype=dt)
    assert_eq(expected, result)
    result = da.add(darr1, darr2, dtype=dt)
    assert_eq(expected, result)