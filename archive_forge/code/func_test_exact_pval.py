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
def test_exact_pval(self):
    x = np.array([1.81, 0.82, 1.56, -0.48, 0.81, 1.28, -1.04, 0.23, -0.75, 0.14])
    y = np.array([0.71, 0.65, -0.2, 0.85, -1.1, -0.45, -0.84, -0.24, -0.68, -0.76])
    _, p = stats.wilcoxon(x, y, alternative='two-sided', mode='exact')
    assert_almost_equal(p, 0.1054688, decimal=6)
    _, p = stats.wilcoxon(x, y, alternative='less', mode='exact')
    assert_almost_equal(p, 0.9580078, decimal=6)
    _, p = stats.wilcoxon(x, y, alternative='greater', mode='exact')
    assert_almost_equal(p, 0.05273438, decimal=6)
    x = np.arange(0, 20) + 0.5
    y = np.arange(20, 0, -1)
    _, p = stats.wilcoxon(x, y, alternative='two-sided', mode='exact')
    assert_almost_equal(p, 0.8694878, decimal=6)
    _, p = stats.wilcoxon(x, y, alternative='less', mode='exact')
    assert_almost_equal(p, 0.4347439, decimal=6)
    _, p = stats.wilcoxon(x, y, alternative='greater', mode='exact')
    assert_almost_equal(p, 0.5795889, decimal=6)