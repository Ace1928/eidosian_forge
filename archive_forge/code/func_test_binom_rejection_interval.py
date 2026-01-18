import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
def test_binom_rejection_interval():
    alpha = 0.05
    nobs = 200
    prop = 12.0 / 20
    alternative = 'smaller'
    ci_low, ci_upp = smprop.binom_test_reject_interval(prop, nobs, alpha=alpha, alternative=alternative)
    assert_equal(ci_upp, nobs)
    pval = smprop.binom_test(ci_low, nobs, prop=prop, alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_low + 1, nobs, prop=prop, alternative=alternative)
    assert_array_less(alpha, pval)
    alternative = 'larger'
    ci_low, ci_upp = smprop.binom_test_reject_interval(prop, nobs, alpha=alpha, alternative=alternative)
    assert_equal(ci_low, 0)
    pval = smprop.binom_test(ci_upp, nobs, prop=prop, alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_upp - 1, nobs, prop=prop, alternative=alternative)
    assert_array_less(alpha, pval)
    alternative = 'two-sided'
    ci_low, ci_upp = smprop.binom_test_reject_interval(prop, nobs, alpha=alpha, alternative=alternative)
    pval = smprop.binom_test(ci_upp, nobs, prop=prop, alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_upp - 1, nobs, prop=prop, alternative=alternative)
    assert_array_less(alpha, pval)
    pval = smprop.binom_test(ci_upp, nobs, prop=prop, alternative=alternative)
    assert_array_less(pval, alpha)
    pval = smprop.binom_test(ci_upp - 1, nobs, prop=prop, alternative=alternative)
    assert_array_less(alpha, pval)