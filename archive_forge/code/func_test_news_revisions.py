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
def test_news_revisions(missing, filter_univariate, tvp):
    mod, res = get_acov_model(missing, filter_univariate, tvp, oos=10)
    params = [] if tvp else mod.start_params
    endog2 = mod.endog.copy()
    endog2[-11] = [0.0, 0.0]
    endog2[-10] = [-0.3, -0.4]
    mod2 = mod.clone(endog2)
    res2 = mod2.smooth(params, return_ssm=True)
    nobs = mod.nobs - 10
    for t in [0, 1, 150, nobs - 1, nobs, nobs + 1, nobs + 9]:
        out = res2.news(res, t=t)
        desired = res2.smoothed_forecasts[..., t] - out.revision_results.smoothed_forecasts[..., t]
        assert_allclose(out.update_impacts, desired, atol=1e-10)
        desired = out.revision_results.smoothed_forecasts[..., t] - res.smoothed_forecasts[..., t]
        assert_allclose(out.revision_impacts, desired, atol=1e-10)