import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_raises, assert_allclose, assert_
from statsmodels import datasets
from statsmodels.tsa.statespace import sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
@pytest.mark.parametrize('use_exog', [True, False])
@pytest.mark.parametrize('trend', ['n', 'c', 't'])
def test_concatenated_predict_varmax(use_exog, trend):
    endog = np.arange(200).reshape(100, 2) * 1.0
    exog = np.ones(100) if use_exog else None
    trend_params = [0.1, 0.2]
    var_params = [0.5, -0.1, 0.0, 0.2]
    exog_params = [1.0, 2.0]
    cov_params = [1.0, 0.0, 1.0]
    params = []
    if trend in ['c', 't']:
        params += trend_params
    params += var_params
    if use_exog:
        params += exog_params
    params += cov_params
    y1 = endog.copy()
    y1[-50:] = np.nan
    mod1 = varmax.VARMAX(y1, order=(1, 0), trend=trend, exog=exog)
    res1 = mod1.smooth(params)
    p1 = res1.get_prediction()
    pr1 = p1.prediction_results
    x2 = exog[:50] if use_exog else None
    mod2 = varmax.VARMAX(endog[:50], order=(1, 0), trend=trend, exog=x2)
    res2 = mod2.smooth(params)
    x2f = exog[50:] if use_exog else None
    p2 = res2.get_prediction(start=0, end=99, exog=x2f)
    pr2 = p2.prediction_results
    attrs = pr1.representation_attributes + pr1.filter_attributes + pr1.smoother_attributes
    for key in attrs:
        assert_allclose(getattr(pr2, key), getattr(pr1, key))