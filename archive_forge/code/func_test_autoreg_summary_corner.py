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
@pytest.mark.parametrize('old_names', [True, False])
def test_autoreg_summary_corner(old_names):
    data = macrodata.load_pandas().data['cpi'].diff().dropna()
    dates = period_range(start='1959Q1', periods=len(data), freq='Q')
    data.index = dates
    warning = FutureWarning if old_names else None
    with pytest_warns(warning):
        res = AutoReg(data, lags=4, old_names=old_names).fit()
    summ = res.summary().as_text()
    assert 'AutoReg(4)' in summ
    assert 'cpi.L4' in summ
    assert '03-31-1960' in summ
    with pytest_warns(warning):
        res = AutoReg(data, lags=0, old_names=old_names).fit()
    summ = res.summary().as_text()
    if old_names:
        assert 'intercept' in summ
    else:
        assert 'const' in summ
    assert 'AutoReg(0)' in summ