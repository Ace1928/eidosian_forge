from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
@pytest.mark.parametrize('index_typ', ['date_range', 'period', 'range', 'irregular'])
def test_forecast_index_types(ses, index_typ):
    nobs = ses.shape[0]
    kwargs = {}
    warning = None
    fcast_index = None
    if index_typ == 'period':
        index = pd.period_range('2000-1-1', periods=nobs + 36, freq='M')
    elif index_typ == 'date_range':
        index = pd.date_range('2000-1-1', periods=nobs + 36, freq=MONTH_END)
    elif index_typ == 'range':
        index = pd.RangeIndex(nobs + 36)
        kwargs['seasonal_periods'] = 12
    elif index_typ == 'irregular':
        rs = np.random.RandomState(0)
        index = pd.Index(np.cumsum(rs.randint(0, 4, size=nobs + 36)))
        warning = ValueWarning
        kwargs['seasonal_periods'] = 12
        fcast_index = pd.RangeIndex(start=1000, stop=1036, step=1)
    if fcast_index is None:
        fcast_index = index[-36:]
    ses = ses.copy()
    ses.index = index[:-36]
    with pytest_warns(warning):
        res = ExponentialSmoothing(ses, trend='add', seasonal='add', initialization_method='heuristic', **kwargs).fit()
    with pytest_warns(warning):
        fcast = res.forecast(36)
    assert isinstance(fcast, pd.Series)
    pd.testing.assert_index_equal(fcast.index, fcast_index)