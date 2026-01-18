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
@pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'), [('two-sided', 1.142629265891, 0.2903950180801), ('less', 0.99629665877411, 0.8545660222131), ('greater', 0.99629665877411, 0.1454339777869)])
def test_against_R_imbalanced(self, alternative, statistic, pvalue):
    rng = np.random.default_rng(5429015622386364034)
    x = rng.random(size=9)
    y = rng.random(size=8)
    res = stats.bws_test(x, y, alternative=alternative)
    assert_allclose(res.statistic, statistic, rtol=1e-13)
    assert_allclose(res.pvalue, pvalue, atol=0.01, rtol=0.1)