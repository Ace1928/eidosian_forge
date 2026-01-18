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
@pytest.mark.smoke
def test_ar_select_order_smoke():
    data = sunspots.load().data['SUNACTIVITY']
    ar_select_order(data, 4, glob=True, trend='n')
    ar_select_order(data, 4, glob=False, trend='n')
    ar_select_order(data, 4, seasonal=True, period=12)
    ar_select_order(data, 4, seasonal=False)
    ar_select_order(data, 4, glob=True)
    ar_select_order(data, 4, glob=True, seasonal=True, period=12)