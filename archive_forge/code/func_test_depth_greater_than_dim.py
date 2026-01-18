from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_depth_greater_than_dim():
    a = np.arange(144).reshape(12, 12)
    darr = da.from_array(a, chunks=(3, 5))
    depth = {0: 13, 1: 4}
    with pytest.raises(ValueError, match='The overlapping depth'):
        overlap(darr, depth=depth, boundary=1)