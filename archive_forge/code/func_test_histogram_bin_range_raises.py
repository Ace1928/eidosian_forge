from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('bins, hist_range', [(None, None), (10, None), (10, 1), (None, (1, 10)), (10, [0, 1, 2]), (10, [0]), (10, np.array([[0, 1]])), (10, da.array([[0, 1]])), ([[0, 1, 2]], None), (np.array([[0, 1, 2]]), None), (da.array([[0, 1, 2]]), None)])
def test_histogram_bin_range_raises(bins, hist_range):
    data = da.random.default_rng().random(10, chunks=2)
    with pytest.raises((ValueError, TypeError)) as info:
        da.histogram(data, bins=bins, range=hist_range)
    err_msg = str(info.value)
    assert 'bins' in err_msg or 'range' in err_msg