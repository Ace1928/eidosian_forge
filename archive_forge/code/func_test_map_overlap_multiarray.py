from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_multiarray():
    x = da.arange(10, chunks=5)
    y = da.arange(10, chunks=5)
    z = da.map_overlap(lambda x, y: x + y, x, y, depth=1, boundary='none')
    assert_eq(z, 2 * np.arange(10))
    x = da.arange(10, chunks=(2, 3, 5))
    y = da.arange(10, chunks=(5, 3, 2))
    z = da.map_overlap(lambda x, y: x + y, x, y, depth=1, boundary='none')
    assert z.chunks == ((2, 3, 3, 2),)
    assert_eq(z, 2 * np.arange(10))
    x = da.arange(10, chunks=(10,))
    y = da.arange(10, chunks=(4, 4, 2))
    z = da.map_overlap(lambda x, y: x + y, x, y, depth=1, boundary='none')
    assert z.chunks == ((4, 4, 2),)
    assert_eq(z, 2 * np.arange(10))
    x = da.arange(10, chunks=(10,))
    y = da.arange(10).reshape(1, 10).rechunk((1, (4, 4, 2)))
    z = da.map_overlap(lambda x, y: x + y, x, y, depth=1, boundary='none')
    assert z.chunks == ((1,), (4, 4, 2))
    assert z.shape == (1, 10)
    assert_eq(z, 2 * np.arange(10)[np.newaxis])