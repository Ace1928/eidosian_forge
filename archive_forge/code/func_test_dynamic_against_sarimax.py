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
def test_dynamic_against_sarimax():
    rs = np.random.RandomState(12345678)
    e = rs.standard_normal(1001)
    y = np.empty(1001)
    y[0] = e[0] * np.sqrt(1.0 / (1 - 0.9 ** 2))
    for i in range(1, 1001):
        y[i] = 0.9 * y[i - 1] + e[i]
    smod = SARIMAX(y, order=(1, 0, 0), trend='c')
    sres = smod.fit(disp=False, iprint=-1)
    mod = AutoReg(y, 1)
    spred = sres.predict(900, 1100)
    pred = mod.predict(sres.params[:2], 900, 1100)
    assert_allclose(spred, pred)
    spred = sres.predict(900, 1100, dynamic=True)
    pred = mod.predict(sres.params[:2], 900, 1100, dynamic=True)
    assert_allclose(spred, pred)
    spred = sres.predict(900, 1100, dynamic=50)
    pred = mod.predict(sres.params[:2], 900, 1100, dynamic=50)
    assert_allclose(spred, pred)