import warnings
import pytest
import inspect
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.nanfunctions import _nan_mask, _replace_nan
from numpy.testing import (
def test_multiple_percentiles(self):
    perc = [50, 100]
    mat = np.ones((4, 3))
    nan_mat = np.nan * mat
    large_mat = np.ones((3, 4, 5))
    large_mat[:, 0:2:4, :] = 0
    large_mat[:, :, 3:] *= 2
    for axis in [None, 0, 1]:
        for keepdim in [False, True]:
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'All-NaN slice encountered')
                val = np.percentile(mat, perc, axis=axis, keepdims=keepdim)
                nan_val = np.nanpercentile(nan_mat, perc, axis=axis, keepdims=keepdim)
                assert_equal(nan_val.shape, val.shape)
                val = np.percentile(large_mat, perc, axis=axis, keepdims=keepdim)
                nan_val = np.nanpercentile(large_mat, perc, axis=axis, keepdims=keepdim)
                assert_equal(nan_val, val)
    megamat = np.ones((3, 4, 5, 6))
    assert_equal(np.nanpercentile(megamat, perc, axis=(1, 2)).shape, (2, 3, 6))