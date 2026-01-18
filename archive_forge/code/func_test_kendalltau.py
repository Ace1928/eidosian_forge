import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def test_kendalltau():
    variants = ('b', 'c')
    x = [5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 6, 3, 1, 8, 7, 4]
    expected = (0.0, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    x = [0, 5, 2, 1, 3, 6, 4, 7, 8]
    y = [5, 2, 0, 6, 3, 1, 8, 7, 4]
    expected = (0.0, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    x = [5, 2, 1, 3, 6, 4, 7]
    y = [5, 2, 6, 3, 1, 7, 4]
    expected = (-0.14285714286, 0.77261904762)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    x = [2, 1, 3, 6, 4, 7, 8]
    y = [2, 6, 3, 1, 8, 7, 4]
    expected = (0.047619047619, 1.0)
    for taux in variants:
        res = stats.kendalltau(x, y)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    x = np.arange(10)
    y = np.arange(10)
    expected = (1.0, 5.511463844797e-07)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    b = y[1]
    y[1] = y[2]
    y[2] = b
    expected = (0.9555555555555556, 5.511463844797e-06)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    b = y[5]
    y[5] = y[6]
    y[6] = b
    expected = (0.9111111111111111, 2.97619047619e-05)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    x = np.arange(10)
    y = np.arange(10)[::-1]
    expected = (-1.0, 5.511463844797e-07)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    b = y[1]
    y[1] = y[2]
    y[2] = b
    expected = (-0.9555555555555556, 5.511463844797e-06)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    b = y[5]
    y[5] = y[6]
    y[6] = b
    expected = (-0.9111111111111111, 2.97619047619e-05)
    for taux in variants:
        res = stats.kendalltau(x, y, variant=taux)
        assert_approx_equal(res[0], expected[0])
        assert_approx_equal(res[1], expected[1])
    x = array([1, 2, 2, 4, 4, 6, 6, 8, 9, 9])
    y = array([1, 2, 4, 4, 4, 4, 8, 8, 8, 10])
    expected = 0.85895569
    assert_approx_equal(stats.kendalltau(x, y, variant='b')[0], expected)
    expected = 0.825
    assert_approx_equal(stats.kendalltau(x, y, variant='c')[0], expected)
    y[2] = y[1]
    assert_raises(ValueError, stats.kendalltau, x, y, method='exact')
    assert_raises(ValueError, stats.kendalltau, x, y, method='banana')
    assert_raises(ValueError, stats.kendalltau, x, y, variant='rms')
    x1 = [12, 2, 1, 12, 2]
    x2 = [1, 4, 7, 1, 0]
    expected = (-0.47140452079103173, 0.2827454599327748)
    res = stats.kendalltau(x1, x2)
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], expected[1])
    attributes = ('correlation', 'pvalue')
    for taux in variants:
        res = stats.kendalltau(x1, x2, variant=taux)
        check_named_results(res, attributes)
        assert_equal(res.correlation, res.statistic)
    for taux in variants:
        assert_equal(stats.kendalltau([2, 2, 2], [2, 2, 2], variant=taux), (np.nan, np.nan))
        assert_equal(stats.kendalltau([2, 0, 2], [2, 2, 2], variant=taux), (np.nan, np.nan))
        assert_equal(stats.kendalltau([2, 2, 2], [2, 0, 2], variant=taux), (np.nan, np.nan))
    assert_equal(stats.kendalltau([], []), (np.nan, np.nan))
    np.random.seed(7546)
    x = np.array([np.random.normal(loc=1, scale=1, size=500), np.random.normal(loc=1, scale=1, size=500)])
    corr = [[1.0, 0.3], [0.3, 1.0]]
    x = np.dot(np.linalg.cholesky(corr), x)
    expected = (0.19291382765531062, 1.1337095377742629e-10)
    res = stats.kendalltau(x[0], x[1])
    assert_approx_equal(res[0], expected[0])
    assert_approx_equal(res[1], expected[1])
    assert_approx_equal(stats.kendalltau([1, 1, 2], [1, 1, 2], variant='b')[0], 1.0)
    assert_approx_equal(stats.kendalltau([1, 1, 2], [1, 1, 2], variant='c')[0], 0.88888888)
    x = np.arange(10.0)
    x[9] = np.nan
    assert_array_equal(stats.kendalltau(x, x), (np.nan, np.nan))
    assert_allclose(stats.kendalltau(x, x, nan_policy='omit'), (1.0, 5.5114638e-06), rtol=1e-06)
    assert_allclose(stats.kendalltau(x, x, nan_policy='omit', method='asymptotic'), (1.0, 0.00017455009626808976), rtol=1e-06)
    assert_raises(ValueError, stats.kendalltau, x, x, nan_policy='raise')
    assert_raises(ValueError, stats.kendalltau, x, x, nan_policy='foobar')
    x = np.arange(10.0)
    y = np.arange(20.0)
    assert_raises(ValueError, stats.kendalltau, x, y)
    tau, p_value = stats.kendalltau([], [])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    tau, p_value = stats.kendalltau([0], [0])
    assert_equal(np.nan, tau)
    assert_equal(np.nan, p_value)
    x = np.arange(2000, dtype=float)
    x = np.ma.masked_greater(x, 1995)
    y = np.arange(2000, dtype=float)
    y = np.concatenate((y[1000:], y[:1000]))
    assert_(np.isfinite(stats.kendalltau(x, y)[1]))