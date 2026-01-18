import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.kalman_filter import (
def test_invalid_fittedvalues_resid_predict():
    endog = dta['infl'].iloc[:20]
    mod = sarimax.SARIMAX(endog, order=(1, 0, 0), concentrate_scale=True)
    res = mod.filter([0], conserve_memory=MEMORY_NO_FORECAST_MEAN)
    assert_equal(res.filter_results.conserve_memory, MEMORY_NO_FORECAST_MEAN)
    message = 'In-sample prediction is not available if memory conservation has been used to avoid storing forecast means.'
    with pytest.raises(ValueError, match=message):
        res.predict()
    with pytest.raises(ValueError, match=message):
        res.get_prediction()
    options = [MEMORY_NO_PREDICTED_MEAN, MEMORY_NO_PREDICTED_COV, MEMORY_NO_PREDICTED]
    for option in options:
        res = mod.filter([0], conserve_memory=option)
        assert_equal(res.filter_results.conserve_memory, option)
        message = 'In-sample dynamic prediction is not available if memory conservation has been used to avoid storing forecasted or predicted state means or covariances.'
        with pytest.raises(ValueError, match=message):
            res.predict(dynamic=True)
        with pytest.raises(ValueError, match=message):
            res.predict(start=endog.index[10], dynamic=True)