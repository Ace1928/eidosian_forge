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
@pytest.mark.parametrize('alternative, pval, ci_high', [('less', 0.05631351, 0.2588656), ('greater', 1.0, 1.0), ('two-sided', 0.07604122, 0.3084971)])
def test_confidence_interval_exact_k0(self, alternative, pval, ci_high):
    res = stats.binomtest(0, 10, p=0.25, alternative=alternative)
    assert_allclose(res.pvalue, pval, rtol=1e-06)
    ci = res.proportion_ci(confidence_level=0.95)
    assert_equal(ci.low, 0.0)
    assert_allclose(ci.high, ci_high, rtol=1e-06)