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
@pytest.mark.parametrize('method', ['asymptotic', 'exact'])
def test_gh_12837_11113(self, method):
    np.random.seed(0)
    axis = -3
    m, n = (7, 10)
    x = np.random.rand(m, 3, 8)
    y = np.random.rand(6, n, 1, 8) + 0.1
    res = mannwhitneyu(x, y, method=method, axis=axis)
    shape = (6, 3, 8)
    assert res.pvalue.shape == shape
    assert res.statistic.shape == shape
    x, y = (np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1))
    x = x[None, ...]
    assert x.ndim == y.ndim
    x = np.broadcast_to(x, shape + (m,))
    y = np.broadcast_to(y, shape + (n,))
    assert x.shape[:-1] == shape
    assert y.shape[:-1] == shape
    statistics = np.zeros(shape)
    pvalues = np.zeros(shape)
    for indices in product(*[range(i) for i in shape]):
        xi = x[indices]
        yi = y[indices]
        temp = mannwhitneyu(xi, yi, method=method)
        statistics[indices] = temp.statistic
        pvalues[indices] = temp.pvalue
    np.testing.assert_equal(res.pvalue, pvalues)
    np.testing.assert_equal(res.statistic, statistics)