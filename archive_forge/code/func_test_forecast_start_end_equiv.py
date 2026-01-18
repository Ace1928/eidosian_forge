from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import datetime as dt
from itertools import product
from typing import NamedTuple, Union
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
from pandas import Index, Series, date_range, period_range
from pandas.testing import assert_series_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import SpecificationWarning, ValueWarning
from statsmodels.tools.tools import Bunch
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tests.results import results_ar
@pytest.mark.parametrize('dynamic', [True, False])
def test_forecast_start_end_equiv(dynamic):
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    effects = 10 * np.cos(np.arange(12) / 11 * 2 * np.pi)
    for i in range(1, 1001):
        y[i] = 10 + 0.9 * y[i - 1] + e[i] + effects[i % 12]
    ys = pd.Series(y, index=pd.date_range(dt.datetime(1950, 1, 1), periods=1001, freq=MONTH_END))
    mod = AutoReg(ys, 1, seasonal=True)
    res = mod.fit()
    pred_int = res.predict(1000, 1020, dynamic=dynamic)
    dates = pd.date_range(dt.datetime(1950, 1, 1), periods=1021, freq=MONTH_END)
    pred_dates = res.predict(dates[1000], dates[1020], dynamic=dynamic)
    assert_series_equal(pred_int, pred_dates)