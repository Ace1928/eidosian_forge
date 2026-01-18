from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
def reduction_0d_test(da_func, darr, np_func, narr):
    expected = np_func(narr)
    actual = da_func(darr)
    assert_eq(actual, expected)
    assert_eq(da_func(narr), expected)
    assert actual.size == 1