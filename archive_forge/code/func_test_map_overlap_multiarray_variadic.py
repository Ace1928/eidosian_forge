from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_multiarray_variadic():
    xs = [da.ones((12, 1, 1), chunks=((12,), 1, 1)), da.ones((12, 8, 1), chunks=((8, 4), 8, 1)), da.ones((12, 8, 4), chunks=((4, 8), 8, 4))]

    def func(*args):
        return np.array([sum((x.size for x in args))])
    x = da.map_overlap(func, *xs, chunks=(1,), depth=1, trim=False, drop_axis=[1, 2], boundary='reflect')
    size_per_slice = sum((np.pad(x[:4], 1, mode='constant').size for x in xs))
    assert x.shape == (3,)
    assert all(x.compute() == size_per_slice)