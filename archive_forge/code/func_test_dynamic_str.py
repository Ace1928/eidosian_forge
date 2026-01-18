import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_dynamic_str():
    data = results_sarimax.wpi1_stationary['data']
    index = pd.date_range('1980-1-1', freq='MS', periods=len(data))
    series = pd.Series(data, index=index)
    mod = sarimax.SARIMAX(series, order=(1, 1, 0), trend='c')
    res = mod.fit()
    dynamic = index[-12]
    desired = res.get_prediction(index[-24], dynamic=12)
    actual = res.get_prediction(index[-24], dynamic=dynamic)
    assert_allclose(actual.predicted_mean, desired.predicted_mean)
    actual = res.get_prediction(index[-24], dynamic=dynamic.to_pydatetime())
    assert_allclose(actual.predicted_mean, desired.predicted_mean)
    actual = res.get_prediction(index[-24], dynamic=dynamic.strftime('%Y-%m-%d'))
    assert_allclose(actual.predicted_mean, desired.predicted_mean)