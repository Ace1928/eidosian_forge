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
@pytest.mark.matplotlib
@pytest.mark.smoke
def test_autoreg_smoke_plots(plot_data, close_figures):
    from matplotlib.figure import Figure
    mod = AutoReg(plot_data.endog, plot_data.lags, trend=plot_data.trend, seasonal=plot_data.seasonal, exog=plot_data.exog, hold_back=plot_data.hold_back, period=plot_data.period, missing=plot_data.missing)
    res = mod.fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, Figure)
    if plot_data.exog is None:
        fig = res.plot_predict(end=300)
        assert isinstance(fig, Figure)
        fig = res.plot_predict(end=300, alpha=None, in_sample=False)
        assert isinstance(fig, Figure)
    assert isinstance(res.summary(), Summary)