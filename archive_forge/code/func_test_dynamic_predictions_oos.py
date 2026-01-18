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
def test_dynamic_predictions_oos(ar2):
    mod = AutoReg(ar2, 2, trend='c')
    res = mod.fit()
    d25_end = res.predict(dynamic=25, end=61)
    s10_d15_end = res.predict(start=10, dynamic=15, end=61)
    end = ar2.index[-1] + 12 * (ar2.index[-1] - ar2.index[-2])
    sd_index_end = res.predict(start=ar2.index[10], dynamic=ar2.index[25], end=end)
    assert_allclose(s10_d15_end, sd_index_end)
    assert_allclose(d25_end[25:], sd_index_end[15:])
    reference = [np.nan, np.nan]
    p = np.asarray(res.params)
    for i in range(2, d25_end.shape[0]):
        if i < ar2.shape[0]:
            lag1 = ar2.iloc[i - 1]
            lag2 = ar2.iloc[i - 2]
        if i > 25:
            lag1 = reference[i - 1]
        if i > 26:
            lag2 = reference[i - 2]
        reference.append(p[0] + p[1] * lag1 + p[2] * lag2)
    expected = pd.Series(reference, index=d25_end.index)
    assert_allclose(expected, d25_end)