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
@pytest.mark.parametrize('weights', [np.array([1, 2, 1, 0.5, 1], dtype=np.float32), np.array([1, 2, 1, 0, 1], dtype=np.int32)])
def test_bincount_with_weights(weights):
    x = np.array([2, 1, 5, 2, 1])
    d = da.from_array(x, chunks=2)
    dweights = da.from_array(weights, chunks=2)
    e = da.bincount(d, weights=dweights, minlength=6)
    assert_eq(e, np.bincount(x, weights=dweights.compute(), minlength=6))
    assert same_keys(da.bincount(d, weights=dweights, minlength=6), e)