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
@pytest.fixture(scope='module', params=params, ids=ids)
def ols_autoreg_result(request):
    ar, seasonal, trend, exog, cov_type = request.param
    y, x, endog, exog = gen_ols_regressors(ar, seasonal, trend, exog)
    ar_mod = AutoReg(y, ar, seasonal=seasonal, trend=trend, exog=x)
    ar_res = ar_mod.fit(cov_type=cov_type)
    ols = OLS(endog, exog)
    ols_res = ols.fit(cov_type=cov_type, use_t=False)
    return (ar_res, ols_res)