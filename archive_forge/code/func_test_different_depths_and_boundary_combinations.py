from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('depth', [{0: 0, 1: 0}, {0: 4, 1: 0}, {0: 5, 1: 5}, {0: 8, 1: 7}])
def test_different_depths_and_boundary_combinations(depth):
    expected = np.arange(100).reshape(10, 10)
    darr = da.from_array(expected, chunks=(5, 2))
    reflected = overlap(darr, depth=depth, boundary='reflect')
    nearest = overlap(darr, depth=depth, boundary='nearest')
    periodic = overlap(darr, depth=depth, boundary='periodic')
    constant = overlap(darr, depth=depth, boundary=42)
    result = trim_internal(reflected, depth, boundary='reflect')
    assert_array_equal(result, expected)
    result = trim_internal(nearest, depth, boundary='nearest')
    assert_array_equal(result, expected)
    result = trim_internal(periodic, depth, boundary='periodic')
    assert_array_equal(result, expected)
    result = trim_internal(constant, depth, boundary=42)
    assert_array_equal(result, expected)