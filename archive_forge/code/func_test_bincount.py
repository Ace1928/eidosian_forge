from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def test_bincount():
    x = np.array([2, 1, 5, 2, 1])
    d = da.from_array(x, chunks=2)
    e = da.bincount(d, minlength=6)
    assert_eq(e, np.bincount(x, minlength=6))
    assert same_keys(da.bincount(d, minlength=6), e)
    assert e.shape == (6,)
    assert e.chunks == ((6,),)
    assert da.bincount(d, minlength=6).name != da.bincount(d, minlength=7).name
    assert da.bincount(d, minlength=6).name == da.bincount(d, minlength=6).name
    expected_output = np.array([0, 2, 2, 0, 0, 1], dtype=e.dtype)
    assert_eq(e[0:], expected_output)