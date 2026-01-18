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
def test_autoreg_forecast_period_index():
    pi = pd.period_range('1990-1-1', periods=524, freq='M')
    y = np.random.RandomState(0).standard_normal(500)
    ys = pd.Series(y, index=pi[:500], name='y')
    mod = AutoReg(ys, 3, seasonal=True)
    res = mod.fit()
    fcast = res.forecast(24)
    assert isinstance(fcast.index, pd.PeriodIndex)
    pd.testing.assert_index_equal(fcast.index, pi[-24:])