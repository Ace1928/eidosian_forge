import warnings
import sys
from functools import partial
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_array_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import re
from scipy import optimize, stats, special
from scipy.stats._morestats import _abw_state, _get_As_weibull, _Avals_weibull
from .common_tests import check_named_results
from .._hypotests import _get_wilcoxon_distr, _get_wilcoxon_distr2
from scipy.stats._binomtest import _binary_search_for_binom_tst
from scipy.stats._distr_params import distcont
def test_fixed_lmbda(self):
    rng = np.random.RandomState(12345)
    x = _old_loggamma_rvs(5, size=50, random_state=rng) + 5
    assert np.all(x > 0)
    xt = stats.yeojohnson(x, lmbda=1)
    assert_allclose(xt, x)
    xt = stats.yeojohnson(x, lmbda=-1)
    assert_allclose(xt, 1 - 1 / (x + 1))
    xt = stats.yeojohnson(x, lmbda=0)
    assert_allclose(xt, np.log(x + 1))
    xt = stats.yeojohnson(x, lmbda=1)
    assert_allclose(xt, x)
    x = _old_loggamma_rvs(5, size=50, random_state=rng) - 5
    assert np.all(x < 0)
    xt = stats.yeojohnson(x, lmbda=2)
    assert_allclose(xt, -np.log(-x + 1))
    xt = stats.yeojohnson(x, lmbda=1)
    assert_allclose(xt, x)
    xt = stats.yeojohnson(x, lmbda=3)
    assert_allclose(xt, 1 / (-x + 1) - 1)
    x = _old_loggamma_rvs(5, size=50, random_state=rng) - 2
    assert not np.all(x < 0)
    assert not np.all(x >= 0)
    pos = x >= 0
    xt = stats.yeojohnson(x, lmbda=1)
    assert_allclose(xt[pos], x[pos])
    xt = stats.yeojohnson(x, lmbda=-1)
    assert_allclose(xt[pos], 1 - 1 / (x[pos] + 1))
    xt = stats.yeojohnson(x, lmbda=0)
    assert_allclose(xt[pos], np.log(x[pos] + 1))
    xt = stats.yeojohnson(x, lmbda=1)
    assert_allclose(xt[pos], x[pos])
    neg = ~pos
    xt = stats.yeojohnson(x, lmbda=2)
    assert_allclose(xt[neg], -np.log(-x[neg] + 1))
    xt = stats.yeojohnson(x, lmbda=1)
    assert_allclose(xt[neg], x[neg])
    xt = stats.yeojohnson(x, lmbda=3)
    assert_allclose(xt[neg], 1 / (-x[neg] + 1) - 1)