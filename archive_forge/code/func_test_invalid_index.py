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
def test_invalid_index(reset_randomstate):
    y = np.random.standard_normal(12 * 200)
    df_y = pd.DataFrame(data=y)
    df_y.index.freq = 'd'
    model = ExponentialSmoothing(df_y, seasonal_periods=12, trend='add', seasonal='add', initialization_method='heuristic')
    fitted = model.fit(optimized=True, use_brute=True)
    fcast = fitted.forecast(steps=157200)
    assert fcast.shape[0] == 157200
    index = pd.date_range('2020-01-01', periods=2 * y.shape[0])
    index = np.random.choice(index, size=df_y.shape[0], replace=False)
    index = sorted(index)
    df_y.index = index
    assert isinstance(df_y.index, pd.DatetimeIndex)
    assert df_y.index.freq is None
    assert df_y.index.inferred_freq is None
    with pytest.warns(ValueWarning, match='A date index has been provided'):
        model = ExponentialSmoothing(df_y, seasonal_periods=12, trend='add', seasonal='add', initialization_method='heuristic')
    fitted = model.fit(optimized=True, use_brute=True)
    with pytest.warns(ValueWarning, match='No supported'):
        fitted.forecast(steps=157200)