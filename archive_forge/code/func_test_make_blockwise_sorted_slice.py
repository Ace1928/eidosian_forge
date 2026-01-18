from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_make_blockwise_sorted_slice():
    x = da.arange(8, chunks=4)
    index = np.array([6, 0, 4, 2, 7, 1, 5, 3])
    a, b = make_block_sorted_slices(index, x.chunks)
    index2 = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    index3 = np.array([3, 0, 2, 1, 7, 4, 6, 5])
    np.testing.assert_array_equal(a, index2)
    np.testing.assert_array_equal(b, index3)