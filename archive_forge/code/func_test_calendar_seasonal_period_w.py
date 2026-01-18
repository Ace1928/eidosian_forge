from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_calendar_seasonal_period_w():
    period = 'W'
    index = pd.date_range('2000-01-03', freq='h', periods=600)
    cs = CalendarSeasonality('h', period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 168] == 1.0
    index = pd.date_range('2000-01-03', freq='B', periods=600)
    cs = CalendarSeasonality('B', period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 5] == 1.0
    index = pd.date_range('2000-01-03', freq='D', periods=600)
    cs = CalendarSeasonality('D', period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 7] == 1.0