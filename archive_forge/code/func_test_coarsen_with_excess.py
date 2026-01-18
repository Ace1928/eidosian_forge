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
def test_coarsen_with_excess():
    x = da.arange(10, chunks=5)
    assert_eq(da.coarsen(np.min, x, {0: 5}, trim_excess=True), np.array([0, 5]))
    assert_eq(da.coarsen(np.sum, x, {0: 3}, trim_excess=True), np.array([0 + 1 + 2, 3 + 4 + 5, 6 + 7 + 8]))