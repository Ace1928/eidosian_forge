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
@pytest.mark.parametrize('not_implemented', [True, False])
def test_no_var_pred(sunspots, not_implemented):

    class DummyPred:

        def __init__(self, predicted_mean, row_labels):
            self.predicted_mean = predicted_mean
            self.row_labels = row_labels

            def f():
                raise NotImplementedError
            if not_implemented:
                self.forecast = property(f)

    class DummyRes:

        def __init__(self, res):
            self._res = res

        def forecast(self, *args, **kwargs):
            return self._res.forecast(*args, **kwargs)

        def get_prediction(self, *args, **kwargs):
            pred = self._res.get_prediction(*args, **kwargs)
            return DummyPred(pred.predicted_mean, pred.row_labels)

    class DummyMod:

        def __init__(self, y):
            self._mod = ARIMA(y)

        def fit(self, *args, **kwargs):
            res = self._mod.fit(*args, **kwargs)
            return DummyRes(res)
    stl_mod = STLForecast(sunspots, model=DummyMod, period=11)
    stl_res = stl_mod.fit()
    with pytest.warns(UserWarning, match='The variance of'):
        pred = stl_res.get_prediction()
    assert np.all(np.isnan(pred.var_pred_mean))