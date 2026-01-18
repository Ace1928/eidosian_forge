from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
def test_map_overlap_deprecated_signature():

    def func(x):
        return np.array(x.sum())
    x = da.ones(3)
    with pytest.warns(FutureWarning):
        y = da.map_overlap(x, func, 0, 'reflect', True)
        assert y.compute() == 3
        assert y.shape == (3,)
    with pytest.warns(FutureWarning):
        y = da.map_overlap(x, func, 1, 'reflect', True)
        assert y.compute() == 5
        assert y.shape == (3,)
    with pytest.warns(FutureWarning):
        y = da.map_overlap(x, func, 1, 'reflect', False)
        assert y.compute() == 5
        assert y.shape == (3,)