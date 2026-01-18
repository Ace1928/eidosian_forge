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
@pytest.mark.parametrize('a', [np.arange(11), np.arange(6).reshape((3, 2))])
def test_average_keepdims(a):
    d_a = da.from_array(a, chunks=2)
    da_avg = da.average(d_a, keepdims=True)
    if NUMPY_GE_123:
        np_avg = np.average(a, keepdims=True)
        assert_eq(np_avg, da_avg)