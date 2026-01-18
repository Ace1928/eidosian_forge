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
def test_histogram_bins_range_with_nan_array():
    v = da.from_array(np.array([-2, np.nan, 2]), chunks=1)
    a1, b1 = da.histogram(v, bins=10, range=(-3, 3))
    a2, b2 = np.histogram(v, bins=10, range=(-3, 3))
    assert_eq(a1, a2)
    assert_eq(b1, b2)