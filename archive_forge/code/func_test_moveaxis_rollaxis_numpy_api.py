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
def test_moveaxis_rollaxis_numpy_api():
    a = da.random.default_rng().random((4, 4, 4), chunks=2)
    result = np.moveaxis(a, 2, 0)
    assert isinstance(result, da.Array)
    assert_eq(result, np.moveaxis(a.compute(), 2, 0))
    result = np.rollaxis(a, 2, 0)
    assert isinstance(result, da.Array)
    assert_eq(result, np.rollaxis(a.compute(), 2, 0))