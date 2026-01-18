from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
def test_like_kendalltau(self):
    x = [5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 6, 3, 1, 8, 7, 4]
    expected = (0.0, 1.0)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
    expected = (0.0, 1.0)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x = [5, 2, 1, 3, 6, 4, 7]
    y = [5, 2, 6, 3, 1, 7, 4]
    expected = (-0.14285714285714, 0.63032695315767)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x = np.arange(10)
    y = np.arange(10)
    expected = (1.0, 0)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x = np.arange(10)
    y = np.array([0, 2, 1, 3, 4, 6, 5, 7, 8, 9])
    expected = (0.91111111111111, 0.0)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x = np.arange(10)
    y = np.arange(10)[::-1]
    expected = (-1.0, 0)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x = np.arange(10)
    y = np.array([9, 7, 8, 6, 5, 3, 4, 2, 1, 0])
    expected = (-0.9111111111111111, 0.0)
    res = stats.somersd(x, y)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    x1 = [12, 2, 1, 12, 2]
    x2 = [1, 4, 7, 1, 0]
    expected = (-0.5, 0.30490178817878)
    res = stats.somersd(x1, x2)
    assert_allclose(res.statistic, expected[0], atol=1e-15)
    assert_allclose(res.pvalue, expected[1], atol=1e-15)
    res = stats.somersd([2, 2, 2], [2, 2, 2])
    assert_allclose(res.statistic, np.nan)
    assert_allclose(res.pvalue, np.nan)
    res = stats.somersd([2, 0, 2], [2, 2, 2])
    assert_allclose(res.statistic, np.nan)
    assert_allclose(res.pvalue, np.nan)
    res = stats.somersd([2, 2, 2], [2, 0, 2])
    assert_allclose(res.statistic, np.nan)
    assert_allclose(res.pvalue, np.nan)
    res = stats.somersd([0], [0])
    assert_allclose(res.statistic, np.nan)
    assert_allclose(res.pvalue, np.nan)
    res = stats.somersd([], [])
    assert_allclose(res.statistic, np.nan)
    assert_allclose(res.pvalue, np.nan)
    x = np.arange(10.0)
    y = np.arange(20.0)
    assert_raises(ValueError, stats.somersd, x, y)