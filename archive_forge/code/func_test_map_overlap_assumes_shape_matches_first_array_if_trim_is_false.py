from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_assumes_shape_matches_first_array_if_trim_is_false():
    x1 = da.ones((10,), chunks=(5, 5))
    x2 = x1.rechunk(10)

    def oversum(x):
        return x[2:-2]
    z1 = da.map_overlap(oversum, x1, depth=2, trim=False, boundary='none')
    assert z1.shape == (10,)
    z2 = da.map_overlap(oversum, x2, depth=2, trim=False, boundary='none')
    assert z2.shape == (10,)