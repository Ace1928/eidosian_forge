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
@pytest.mark.parametrize('alternative, pval, ci_low, ci_high', [('less', 0.148831050443, 0.0, 0.2772002496709138), ('greater', 0.9004695898947, 0.1366613252458672, 1.0), ('two-sided', 0.2983720970096, 0.1266555521019559, 0.2918426890886281)])
def test_confidence_intervals1(self, alternative, pval, ci_low, ci_high):
    res = stats.binomtest(20, n=100, p=0.25, alternative=alternative)
    assert_allclose(res.pvalue, pval, rtol=1e-12)
    assert_equal(res.statistic, 0.2)
    ci = res.proportion_ci(confidence_level=0.95)
    assert_allclose((ci.low, ci.high), (ci_low, ci_high), rtol=1e-12)