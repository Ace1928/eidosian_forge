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
@pytest.mark.parametrize(('use_continuity', 'alternative', 'method', 'pvalue_exp'), cases_9184)
def test_gh_9184(self, use_continuity, alternative, method, pvalue_exp):
    statistic_exp = 35
    x = (0.8, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46)
    y = (1.15, 0.88, 0.9, 0.74, 1.21)
    res = mannwhitneyu(x, y, use_continuity=use_continuity, alternative=alternative, method=method)
    assert_equal(res.statistic, statistic_exp)
    assert_allclose(res.pvalue, pvalue_exp)