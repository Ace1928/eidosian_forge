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
def test_histogramdd_edges():
    data = da.random.default_rng().random(size=(10, 3), chunks=(5, 3))
    edges = [np.array([0.1, 0.3, 0.8, 1.0]), np.array([0.2, 0.3, 0.8, 0.9]), np.array([0.1, 0.5, 0.7])]
    a1, b1 = da.histogramdd(data, bins=edges)
    a2, b2 = np.histogramdd(data.compute(), bins=edges)
    for ib1, ib2 in zip(b1, b2):
        assert_eq(ib1, ib2)
    a1, b1 = da.histogramdd(data, bins=5, range=((0, 1),) * 3)
    a2, b2 = np.histogramdd(data.compute(), bins=5, range=((0, 1),) * 3)
    for ib1, ib2 in zip(b1, b2):
        assert_eq(ib1, ib2)