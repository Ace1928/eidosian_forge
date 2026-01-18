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
def test_iscomplexobj():
    a = da.from_array(np.array([1, 2]), 2)
    assert np.iscomplexobj(a) is False
    a = da.from_array(np.array([1, 2 + 0j]), 2)
    assert np.iscomplexobj(a) is True