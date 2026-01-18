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
def test_contingency_table_with_zero_rows_cols(self):
    N = 100
    shape = (4, 6)
    size = np.prod(shape)
    np.random.seed(0)
    s = stats.multinomial.rvs(N, p=np.ones(size) / size).reshape(shape)
    res = stats.somersd(s)
    s2 = np.insert(s, 2, np.zeros(shape[1]), axis=0)
    res2 = stats.somersd(s2)
    s3 = np.insert(s, 2, np.zeros(shape[0]), axis=1)
    res3 = stats.somersd(s3)
    s4 = np.insert(s2, 2, np.zeros(shape[0] + 1), axis=1)
    res4 = stats.somersd(s4)
    assert_allclose(res.statistic, -0.11698113207547, atol=1e-15)
    assert_allclose(res.statistic, res2.statistic)
    assert_allclose(res.statistic, res3.statistic)
    assert_allclose(res.statistic, res4.statistic)
    assert_allclose(res.pvalue, 0.15637644818815, atol=1e-15)
    assert_allclose(res.pvalue, res2.pvalue)
    assert_allclose(res.pvalue, res3.pvalue)
    assert_allclose(res.pvalue, res4.pvalue)