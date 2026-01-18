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
def test_gh_11355(self):
    x = [1, 2, 3, 4]
    y = [3, 6, 7, 8, 9, 3, 2, 1, 4, 4, 5]
    res1 = mannwhitneyu(x, y)
    y[4] = np.inf
    res2 = mannwhitneyu(x, y)
    assert_equal(res1.statistic, res2.statistic)
    assert_equal(res1.pvalue, res2.pvalue)
    y[4] = np.nan
    res3 = mannwhitneyu(x, y)
    assert_equal(res3.statistic, np.nan)
    assert_equal(res3.pvalue, np.nan)