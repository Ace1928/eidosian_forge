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
def test_histogramdd_weighted_density():
    rng = da.random.default_rng()
    n1, n2 = (1200, 4)
    x = rng.standard_normal(size=(n1, n2), chunks=(200, 4))
    w = rng.uniform(0.5, 1.2, size=(n1,), chunks=200)
    bins = (5, 6, 7, 8)
    ranges = ((-4, 4),) * len(bins)
    a1, b1 = da.histogramdd(x, bins=bins, range=ranges, weights=w, density=True)
    a2, b2 = np.histogramdd(x, bins=bins, range=ranges, weights=w, density=True)
    a3, b3 = da.histogramdd(x, bins=bins, range=ranges, weights=w, normed=True)
    assert_eq(a1, a2)
    assert_eq(a1, a3)