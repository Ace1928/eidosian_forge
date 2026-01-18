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
@pytest.mark.parametrize(('alternative', 'statistic', 'pvalue'), [('two-sided', 1.7510204081633, 0.1264422777777), ('less', -1.7510204081633, 0.05754662004662), ('greater', -1.7510204081633, 0.9424533799534)])
def test_against_R(self, alternative, statistic, pvalue):
    rng = np.random.default_rng(4571775098104213308)
    x, y = rng.random(size=(2, 7))
    res = stats.bws_test(x, y, alternative=alternative)
    assert_allclose(res.statistic, statistic, rtol=1e-13)
    assert_allclose(res.pvalue, pvalue, atol=0.01, rtol=0.1)