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
def test_dot_method():
    x = np.arange(400).reshape((20, 20))
    a = da.from_array(x, chunks=(5, 5))
    y = np.arange(200).reshape((20, 10))
    b = da.from_array(y, chunks=(5, 5))
    assert_eq(a.dot(b), x.dot(y))