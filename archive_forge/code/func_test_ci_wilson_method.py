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
@pytest.mark.parametrize('k, alternative, corr, conf, ci_low, ci_high', [[3, 'two-sided', True, 0.95, 0.08094782, 0.64632928], [3, 'two-sided', True, 0.99, 0.0586329, 0.7169416], [3, 'two-sided', False, 0.95, 0.1077913, 0.6032219], [3, 'two-sided', False, 0.99, 0.07956632, 0.6799753], [3, 'less', True, 0.95, 0.0, 0.6043476], [3, 'less', True, 0.99, 0.0, 0.6901811], [3, 'less', False, 0.95, 0.0, 0.5583002], [3, 'less', False, 0.99, 0.0, 0.6507187], [3, 'greater', True, 0.95, 0.09644904, 1.0], [3, 'greater', True, 0.99, 0.06659141, 1.0], [3, 'greater', False, 0.95, 0.1268766, 1.0], [3, 'greater', False, 0.99, 0.08974147, 1.0], [0, 'two-sided', True, 0.95, 0.0, 0.3445372], [0, 'two-sided', False, 0.95, 0.0, 0.2775328], [0, 'less', True, 0.95, 0.0, 0.2847374], [0, 'less', False, 0.95, 0.0, 0.212942], [0, 'greater', True, 0.95, 0.0, 1.0], [0, 'greater', False, 0.95, 0.0, 1.0], [10, 'two-sided', True, 0.95, 0.6554628, 1.0], [10, 'two-sided', False, 0.95, 0.7224672, 1.0], [10, 'less', True, 0.95, 0.0, 1.0], [10, 'less', False, 0.95, 0.0, 1.0], [10, 'greater', True, 0.95, 0.7152626, 1.0], [10, 'greater', False, 0.95, 0.787058, 1.0]])
def test_ci_wilson_method(self, k, alternative, corr, conf, ci_low, ci_high):
    res = stats.binomtest(k, n=10, p=0.1, alternative=alternative)
    if corr:
        method = 'wilsoncc'
    else:
        method = 'wilson'
    ci = res.proportion_ci(confidence_level=conf, method=method)
    assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-06)