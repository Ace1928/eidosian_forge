from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import statsmodels.datasets
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.statespace.exponential_smoothing import (
@pytest.mark.parametrize('config', MODELS, ids=IDS)
@pytest.mark.parametrize('horizon', [1, 7, 23])
def test_equivalence_forecast(data, config, horizon):
    model, kwargs = config
    stl = STL(data)
    stl_fit = stl.fit()
    resids = data - stl_fit.seasonal
    mod = model(resids, **kwargs)
    fit_kwarg = {}
    if model is ETSModel:
        fit_kwarg['disp'] = False
    res = mod.fit(**fit_kwarg)
    stlf = STLForecast(data, model, model_kwargs=kwargs).fit(fit_kwargs=fit_kwarg)
    seasonal = np.asarray(stl_fit.seasonal)[-12:]
    seasonal = np.tile(seasonal, 1 + horizon // 12)
    fcast = res.forecast(horizon) + seasonal[:horizon]
    actual = stlf.forecast(horizon)
    assert_allclose(actual, fcast, rtol=0.0001)
    if not hasattr(res, 'get_prediction'):
        return
    pred = stlf.get_prediction(data.shape[0], data.shape[0] + horizon - 1)
    assert isinstance(pred, PredictionResults)
    assert_allclose(pred.predicted_mean, fcast, rtol=0.0001)
    half = data.shape[0] // 2
    stlf.get_prediction(half, data.shape[0] + horizon - 1)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=True)
    stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=half // 2)
    if hasattr(data, 'index'):
        loc = data.index[half + half // 2]
        a = stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=loc.strftime('%Y-%m-%d'))
        b = stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=loc.to_pydatetime())
        c = stlf.get_prediction(half, data.shape[0] + horizon - 1, dynamic=loc)
        assert_allclose(a.predicted_mean, b.predicted_mean, rtol=0.0001)
        assert_allclose(a.predicted_mean, c.predicted_mean, rtol=0.0001)