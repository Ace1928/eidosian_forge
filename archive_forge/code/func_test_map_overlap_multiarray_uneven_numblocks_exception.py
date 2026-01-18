from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_multiarray_uneven_numblocks_exception():
    x = da.arange(10, chunks=(10,))
    y = da.arange(10, chunks=(5, 5))
    with pytest.raises(ValueError):
        da.map_overlap(lambda x, y: x + y, x, y, align_arrays=False, boundary='none').compute()