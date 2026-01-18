import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
@pytest.mark.parametrize('missing', ['all', 'partial', 'mixed', None])
@pytest.mark.parametrize('filter_univariate', [True, False])
@pytest.mark.parametrize('tvp', [True, False])
def test_news_invalid(missing, filter_univariate, tvp):
    error_ss = 'This results object has %s and so it does not appear to by an extension of `previous`. Can only compute the news by comparing this results set to previous results objects.'
    mod, res = get_acov_model(missing, filter_univariate, tvp, oos=1)
    params = [] if tvp else mod.start_params
    endog2 = mod.endog.copy()
    endog2[-1] = [0.2, 0.5]
    mod2 = mod.clone(endog2)
    res2_filtered = mod2.filter(params, return_ssm=True)
    res2_smoothed = mod2.smooth(params, return_ssm=True)
    res2_smoothed.news(res, t=mod.nobs - 1)
    msg = 'Cannot compute news without having applied the Kalman smoother first.'
    with pytest.raises(ValueError, match=msg):
        res2_filtered.news(res, t=mod.nobs - 1)
    if tvp:
        msg = 'Cannot compute the impacts of news on periods outside of the sample in time-varying models.'
        with pytest.raises(RuntimeError, match=msg):
            res2_smoothed.news(res, t=mod.nobs + 2)
    mod, res = get_acov_model(missing, filter_univariate, tvp)
    params = [] if tvp else mod.start_params
    endog2 = mod.endog.copy()[:mod.nobs - 1]
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    msg = error_ss % 'fewer observations than `previous`'
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)
    mod2 = sarimax.SARIMAX(np.zeros(mod.nobs))
    res2 = mod2.smooth([0.5, 1.0], return_ssm=True)
    msg = error_ss % 'different state space dimensions than `previous`'
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)
    mod2, res2 = get_acov_model(missing, filter_univariate, not tvp, oos=1)
    if tvp:
        msg = 'time-invariant design while `previous` does not'
    else:
        msg = 'time-varying design while `previous` does not'
    with pytest.raises(ValueError, match=msg):
        res2.news(res, t=mod.nobs - 1)