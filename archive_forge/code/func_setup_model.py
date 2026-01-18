from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
@pytest.fixture(params=ALL_MODELS_AND_DATA, ids=ALL_MODEL_IDS)
def setup_model(request, austourists, oildata, ets_austourists_fit_results_R, ets_oildata_fit_results_R):
    params = request.param
    error, trend, seasonal, damped = params[0:4]
    data = params[4]
    if data == 'austourists':
        data = austourists
        seasonal_periods = 4
        results = ets_austourists_fit_results_R[damped]
    else:
        data = oildata
        seasonal_periods = None
        results = ets_oildata_fit_results_R[damped]
    name = short_model_name(error, trend, seasonal)
    if name not in results:
        pytest.skip(f'model {name} not implemented or not converging in R')
    results_R = results[name]
    params = get_params_from_R(results_R)
    model = ETSModel(data, seasonal_periods=seasonal_periods, error=error, trend=trend, seasonal=seasonal, damped_trend=damped)
    return (model, params, results_R)