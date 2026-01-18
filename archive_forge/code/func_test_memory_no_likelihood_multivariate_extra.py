import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
@pytest.mark.parametrize('univariate', [True, False])
@pytest.mark.parametrize('diffuse', [True, False])
@pytest.mark.parametrize('collapsed', [True, False])
def test_memory_no_likelihood_multivariate_extra(univariate, diffuse, collapsed):
    endog = dta[['infl', 'realint']].iloc[:20].copy()
    endog.iloc[0, 0] = np.nan
    endog.iloc[4:6, :] = np.nan
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True
    if collapsed:
        mod.ssm.filter_collapsed = True
    params = [4, -4.5, 0.8, 0.9, -0.5]
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)
    assert_allclose(res1.llf, res2.llf)