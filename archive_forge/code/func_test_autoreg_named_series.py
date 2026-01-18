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
def test_autoreg_named_series(reset_randomstate, old_names):
    warning = FutureWarning if old_names else None
    dates = period_range(start='2011-1', periods=72, freq='M')
    y = Series(np.random.randn(72), name='foobar', index=dates)
    with pytest_warns(warning):
        results = AutoReg(y, lags=2, old_names=old_names).fit()
    if old_names:
        idx = Index(['intercept', 'foobar.L1', 'foobar.L2'])
    else:
        idx = Index(['const', 'foobar.L1', 'foobar.L2'])
    assert results.params.index.equals(idx)