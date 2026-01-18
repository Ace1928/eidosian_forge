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
def test_histogramdd_seq_of_arrays():
    rng = da.random.default_rng()
    n1 = 800
    x = rng.uniform(size=(n1,), chunks=200)
    y = rng.uniform(size=(n1,), chunks=200)
    bx = [0.0, 0.25, 0.75, 1.0]
    by = [0.0, 0.3, 0.7, 0.8, 1.0]
    a1, b1 = da.histogramdd([x, y], bins=[bx, by])
    a2, b2 = np.histogramdd([x, y], bins=[bx, by])
    a3, b3 = np.histogramdd((x.compute(), y.compute()), bins=[bx, by])
    assert_eq(a1, a2)
    assert_eq(a1, a3)