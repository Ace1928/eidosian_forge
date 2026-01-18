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
def test_binom_test():
    binom_test_less = Holder()
    binom_test_less.statistic = 51
    binom_test_less.parameter = 235
    binom_test_less.p_value = 0.982022657605858
    binom_test_less.conf_int = [0, 0.2659460862574313]
    binom_test_less.estimate = 0.2170212765957447
    binom_test_less.null_value = 1.0 / 6
    binom_test_less.alternative = 'less'
    binom_test_less.method = 'Exact binomial test'
    binom_test_less.data_name = '51 and 235'
    binom_test_greater = Holder()
    binom_test_greater.statistic = 51
    binom_test_greater.parameter = 235
    binom_test_greater.p_value = 0.02654424571169085
    binom_test_greater.conf_int = [0.1735252778065201, 1]
    binom_test_greater.estimate = 0.2170212765957447
    binom_test_greater.null_value = 1.0 / 6
    binom_test_greater.alternative = 'greater'
    binom_test_greater.method = 'Exact binomial test'
    binom_test_greater.data_name = '51 and 235'
    binom_test_2sided = Holder()
    binom_test_2sided.statistic = 51
    binom_test_2sided.parameter = 235
    binom_test_2sided.p_value = 0.0437479701823997
    binom_test_2sided.conf_int = [0.1660633298083073, 0.2752683640289254]
    binom_test_2sided.estimate = 0.2170212765957447
    binom_test_2sided.null_value = 1.0 / 6
    binom_test_2sided.alternative = 'two.sided'
    binom_test_2sided.method = 'Exact binomial test'
    binom_test_2sided.data_name = '51 and 235'
    alltests = [('larger', binom_test_greater), ('smaller', binom_test_less), ('two-sided', binom_test_2sided)]
    for alt, res0 in alltests:
        res = smprop.binom_test(51, 235, prop=1.0 / 6, alternative=alt)
        assert_almost_equal(res, res0.p_value, decimal=13)
    ci_2s = smprop.proportion_confint(51, 235, alpha=0.05, method='beta')
    ci_low, ci_upp = smprop.proportion_confint(51, 235, alpha=0.1, method='beta')
    assert_almost_equal(ci_2s, binom_test_2sided.conf_int, decimal=13)
    assert_almost_equal(ci_upp, binom_test_less.conf_int[1], decimal=13)
    assert_almost_equal(ci_low, binom_test_greater.conf_int[0], decimal=13)