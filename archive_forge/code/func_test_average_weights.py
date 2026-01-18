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
@pytest.mark.parametrize('keepdims', [False, True])
def test_average_weights(keepdims):
    a = np.arange(6).reshape((3, 2))
    d_a = da.from_array(a, chunks=2)
    weights = np.array([0.25, 0.75])
    d_weights = da.from_array(weights, chunks=2)
    da_avg = da.average(d_a, weights=d_weights, axis=1, keepdims=keepdims)
    if NUMPY_GE_123:
        assert_eq(da_avg, np.average(a, weights=weights, axis=1, keepdims=keepdims))
    elif not keepdims:
        assert_eq(da_avg, np.average(a, weights=weights, axis=1))