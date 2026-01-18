from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def test_mixed_mask_nan_1():
    m, n = (3, 20)
    axis = -1
    np.random.seed(0)
    a = np.random.rand(m, n)
    b = np.random.rand(m, n)
    mask_a1 = np.random.rand(m, n) < 0.2
    mask_a2 = np.random.rand(m, n) < 0.1
    mask_b1 = np.random.rand(m, n) < 0.15
    mask_b2 = np.random.rand(m, n) < 0.15
    mask_a1[2, :] = True
    a_nans = a.copy()
    b_nans = b.copy()
    a_nans[mask_a1 | mask_a2] = np.nan
    b_nans[mask_b1 | mask_b2] = np.nan
    a_masked1 = np.ma.masked_array(a, mask=mask_a1)
    b_masked1 = np.ma.masked_array(b, mask=mask_b1)
    a_masked1[mask_a2] = np.nan
    b_masked1[mask_b2] = np.nan
    a_masked2 = np.ma.masked_array(a, mask=mask_a2)
    b_masked2 = np.ma.masked_array(b, mask=mask_b2)
    a_masked2[mask_a1] = np.nan
    b_masked2[mask_b1] = np.nan
    a_masked3 = np.ma.masked_array(a, mask=mask_a1 | mask_a2)
    b_masked3 = np.ma.masked_array(b, mask=mask_b1 | mask_b2)
    res = stats.wilcoxon(a_nans, b_nans, nan_policy='omit', axis=axis)
    res1 = stats.wilcoxon(a_masked1, b_masked1, nan_policy='omit', axis=axis)
    res2 = stats.wilcoxon(a_masked2, b_masked2, nan_policy='omit', axis=axis)
    res3 = stats.wilcoxon(a_masked3, b_masked3, nan_policy='raise', axis=axis)
    res4 = stats.wilcoxon(a_masked3, b_masked3, nan_policy='propagate', axis=axis)
    np.testing.assert_array_equal(res1, res)
    np.testing.assert_array_equal(res2, res)
    np.testing.assert_array_equal(res3, res)
    np.testing.assert_array_equal(res4, res)