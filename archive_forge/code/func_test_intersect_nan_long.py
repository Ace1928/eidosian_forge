from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_intersect_nan_long():
    old_chunks = (tuple([np.nan] * 4), (10,))
    new_chunks = (tuple([np.nan] * 4), (5, 5))
    result = list(intersect_chunks(old_chunks, new_chunks))
    expected = [(((0, slice(0, None, None)), (0, slice(0, 5, None))),), (((0, slice(0, None, None)), (0, slice(5, 10, None))),), (((1, slice(0, None, None)), (0, slice(0, 5, None))),), (((1, slice(0, None, None)), (0, slice(5, 10, None))),), (((2, slice(0, None, None)), (0, slice(0, 5, None))),), (((2, slice(0, None, None)), (0, slice(5, 10, None))),), (((3, slice(0, None, None)), (0, slice(0, 5, None))),), (((3, slice(0, None, None)), (0, slice(5, 10, None))),)]
    assert result == expected