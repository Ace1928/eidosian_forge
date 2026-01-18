from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_constant_boundaries():
    a = np.arange(1 * 9).reshape(1, 9)
    darr = da.from_array(a, chunks=((1,), (2, 2, 2, 3)))
    b = boundaries(darr, {0: 0, 1: 0}, {0: 0, 1: 0})
    assert b.chunks == darr.chunks