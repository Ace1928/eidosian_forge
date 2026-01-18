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
@pytest.mark.parametrize('method', probci_methods)
def test_confint_proportion_ndim(method):
    count = np.arange(6).reshape(2, 3)
    nobs = 10 * np.ones((2, 3))
    count_pd = pd.DataFrame(count)
    nobs_pd = pd.DataFrame(nobs)
    ci_arr = proportion_confint(count, nobs, alpha=0.05, method=method)
    ci_pd = proportion_confint(count_pd, nobs_pd, alpha=0.05, method=method)
    assert_allclose(ci_arr, (ci_pd[0].values, ci_pd[1].values), rtol=1e-13)
    ci12 = proportion_confint(count[1, 2], nobs[1, 2], alpha=0.05, method=method)
    assert_allclose((ci_pd[0].values[1, 2], ci_pd[1].values[1, 2]), ci12, rtol=1e-13)
    assert_allclose((ci_arr[0][1, 2], ci_arr[1][1, 2]), ci12, rtol=1e-13)
    ci_li = proportion_confint(count.tolist(), nobs.tolist(), alpha=0.05, method=method)
    assert_allclose(ci_arr, (ci_li[0], ci_li[1]), rtol=1e-13)
    ci_pds = proportion_confint(count_pd.iloc[0], nobs_pd.iloc[0], alpha=0.05, method=method)
    assert_allclose((ci_pds[0].values, ci_pds[1].values), (ci_pd[0].values[0], ci_pd[1].values[0]), rtol=1e-13)
    ci_arr2 = proportion_confint(count, nobs[1, 2], alpha=0.05, method=method)
    assert_allclose((ci_arr2[0][1, 2], ci_arr[1][1, 2]), ci12, rtol=1e-13)
    ci_arr2 = proportion_confint(count + 0.0001, nobs[1, 2], alpha=0.05, method=method)
    assert_allclose((ci_arr2[0][1, 2], ci_arr[1][1, 2]), ci12, rtol=0.0001)