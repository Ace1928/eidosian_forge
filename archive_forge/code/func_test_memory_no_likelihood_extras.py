import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
@pytest.mark.parametrize('concentrate', [True, False])
@pytest.mark.parametrize('univariate', [True, False])
@pytest.mark.parametrize('diffuse', [True, False])
@pytest.mark.parametrize('timing_init_filtered', [True, False])
def test_memory_no_likelihood_extras(concentrate, univariate, diffuse, timing_init_filtered):
    endog = dta['infl'].iloc[:20].copy()
    endog.iloc[0] = np.nan
    endog.iloc[4:6] = np.nan
    exog = dta['realint'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), exog=exog, concentrate_scale=concentrate)
    if timing_init_filtered:
        mod.timing_init_filtered = True
    if diffuse:
        mod.ssm.initialize_diffuse()
    if univariate:
        mod.ssm.filter_univariate = True
    params = [1.2, 0.85]
    if not concentrate:
        params.append(7.0)
    res1 = mod.filter(params)
    mod.ssm.memory_no_likelihood = True
    res2 = mod.filter(params)
    assert_equal(len(res1.llf_obs), 20)
    assert_equal(res2.llf_obs, None)
    assert_allclose(res1.llf, res2.llf)