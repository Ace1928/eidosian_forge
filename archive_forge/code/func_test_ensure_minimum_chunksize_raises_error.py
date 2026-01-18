from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_ensure_minimum_chunksize_raises_error():
    chunks = (5, 2, 1, 1)
    with pytest.raises(ValueError, match='overlapping depth 10 is larger than'):
        ensure_minimum_chunksize(10, chunks)