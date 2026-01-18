import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
def test_extend_results_exog():
    endog = np.arange(200).reshape(100, 2)
    exog = np.ones(100)
    params = [0.1, 0.2, 0.5, -0.1, 0.0, 0.2, 1.0, 2.0, 1.0, 0.0, 1.0]
    mod1 = varmax.VARMAX(endog, order=(1, 0), trend='t', exog=exog)
    res1 = mod1.smooth(params)
    mod2 = varmax.VARMAX(endog[:50], order=(1, 0), trend='t', exog=exog[:50])
    res2 = mod2.smooth(params)
    res3 = res2.extend(endog[50:], exog=exog[50:])
    assert_allclose(res3.llf_obs, res1.llf_obs[50:])
    for attr in ['filtered_state', 'filtered_state_cov', 'predicted_state', 'predicted_state_cov', 'forecasts', 'forecasts_error', 'forecasts_error_cov', 'standardized_forecasts_error', 'forecasts_error_diffuse_cov', 'predicted_diffuse_state_cov', 'scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothing_error', 'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_measurement_disturbance', 'smoothed_state_disturbance', 'smoothed_measurement_disturbance_cov', 'smoothed_state_disturbance_cov']:
        desired = getattr(res1, attr)
        if desired is not None:
            desired = desired[..., 50:]
        assert_equal(getattr(res3, attr), desired)
    assert_allclose(res3.forecast(10, exog=np.ones(10)), res1.forecast(10, exog=np.ones(10)))