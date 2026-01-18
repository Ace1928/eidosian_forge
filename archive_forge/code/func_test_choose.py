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
def test_choose():
    x = np.random.default_rng().integers(10, size=(15, 16))
    d = da.from_array(x, chunks=(4, 5))
    assert_eq(da.choose(d > 5, [0, d]), np.choose(x > 5, [0, x]))
    assert_eq(da.choose(d > 5, [-d, d]), np.choose(x > 5, [-x, x]))
    index_dask = d > 5
    index_numpy = x > 5
    assert_eq(index_dask.choose([0, d]), index_numpy.choose([0, x]))
    assert_eq(index_dask.choose([-d, d]), index_numpy.choose([-x, x]))