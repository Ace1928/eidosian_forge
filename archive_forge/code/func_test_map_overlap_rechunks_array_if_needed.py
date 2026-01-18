from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_rechunks_array_if_needed():
    expected = np.arange(11)
    x = da.from_array(expected, chunks=5)
    y = x.map_overlap(lambda x: x, depth=2, boundary=0)
    assert all((c >= 2 for c in y.chunks[0]))
    assert_eq(y, expected)