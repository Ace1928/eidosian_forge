from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_escapes_to_map_blocks_when_depth_is_zero():
    x = da.arange(10, chunks=5)
    y = x.map_overlap(lambda x: x + 1, depth=0, boundary='none')
    assert len(y.dask) == 2 * x.numblocks[0]
    assert_eq(y, np.arange(10) + 1)