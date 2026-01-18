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
@pytest.mark.parametrize('case', res_binom)
@pytest.mark.parametrize('method', probci_methods)
def test_confint_proportion(method, case):
    count, nobs = case
    idx = res_binom_methods.index(probci_methods[method])
    res_low = res_binom[case].ci_low[idx]
    res_upp = res_binom[case].ci_upp[idx]
    if np.isnan(res_low) or np.isnan(res_upp):
        pytest.skip('Skipping due to NaN value')
    if (count == 0 or count == nobs) and method == 'jeffreys':
        pytest.skip('Skipping nobs 0 or count and jeffreys')
    if method == 'jeffreys' and nobs == 30:
        pytest.skip('Skipping nobs is 30 and jeffreys due to extreme case problem')
    ci = proportion_confint(count, nobs, alpha=0.05, method=method)
    res_low = max(res_low, 0)
    res_upp = min(res_upp, 1)
    assert_almost_equal(ci, [res_low, res_upp], decimal=6, err_msg=repr(case) + method)