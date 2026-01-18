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
def test_directions(self):
    rng = np.random.default_rng(1520514347193347862)
    x = rng.random(size=5)
    y = x - 1
    res = stats.bws_test(x, y, alternative='greater')
    assert res.statistic > 0
    assert_equal(res.pvalue, 1 / len(res.null_distribution))
    res = stats.bws_test(x, y, alternative='less')
    assert res.statistic > 0
    assert_equal(res.pvalue, 1)
    res = stats.bws_test(y, x, alternative='less')
    assert res.statistic < 0
    assert_equal(res.pvalue, 1 / len(res.null_distribution))
    res = stats.bws_test(y, x, alternative='greater')
    assert res.statistic < 0
    assert_equal(res.pvalue, 1)