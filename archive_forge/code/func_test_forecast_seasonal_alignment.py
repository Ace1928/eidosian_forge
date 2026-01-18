from itertools import product
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.forecasting.theta import ThetaModel
@pytest.mark.parametrize('period', [4, 12])
def test_forecast_seasonal_alignment(data, period):
    res = ThetaModel(data, period=period, deseasonalize=True, use_test=False, difference=False).fit(use_mle=False)
    seasonal = res._seasonal
    comp = res.forecast_components(32)
    index = np.arange(data.shape[0], data.shape[0] + comp.shape[0])
    expected = seasonal[index % period]
    np.testing.assert_allclose(comp.seasonal, expected)