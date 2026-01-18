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
@pytest.fixture(scope='module')
def ar2(request):
    gen = np.random.RandomState(20210623)
    e = gen.standard_normal(52)
    y = 10 * np.ones_like(e)
    for i in range(2, y.shape[0]):
        y[i] = 1 + 0.5 * y[i - 1] + 0.4 * y[i - 2] + e[i]
    index = pd.period_range('2000-01-01', periods=e.shape[0] - 2, freq='M')
    return pd.Series(y[2:], index=index)