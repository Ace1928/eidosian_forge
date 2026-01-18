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
@pytest.mark.parametrize('n', [0, 1, 2])
def test_diff_prepend(n):
    x = np.arange(5) + 1
    a = da.from_array(x, chunks=2)
    assert_eq(da.diff(a, n, prepend=0), np.diff(x, n, prepend=0))
    assert_eq(da.diff(a, n, prepend=[0]), np.diff(x, n, prepend=[0]))
    assert_eq(da.diff(a, n, prepend=[-1, 0]), np.diff(x, n, prepend=[-1, 0]))
    x = np.arange(16).reshape(4, 4)
    a = da.from_array(x, chunks=2)
    assert_eq(da.diff(a, n, axis=1, prepend=0), np.diff(x, n, axis=1, prepend=0))
    assert_eq(da.diff(a, n, axis=1, prepend=[[0], [0], [0], [0]]), np.diff(x, n, axis=1, prepend=[[0], [0], [0], [0]]))
    assert_eq(da.diff(a, n, axis=0, prepend=0), np.diff(x, n, axis=0, prepend=0))
    assert_eq(da.diff(a, n, axis=0, prepend=[[0, 0, 0, 0]]), np.diff(x, n, axis=0, prepend=[[0, 0, 0, 0]]))
    if n > 0:
        with pytest.raises(ValueError):
            da.diff(a, n, prepend=np.zeros((3, 3)))