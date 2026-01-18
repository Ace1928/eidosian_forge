from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('boundary', [None, 'reflect', 'periodic', 'nearest', 'none', 0])
def test_map_overlap_no_depth(boundary):
    x = da.arange(10, chunks=5)
    y = x.map_overlap(lambda i: i, depth=0, boundary=boundary, dtype=x.dtype)
    assert_eq(y, x)