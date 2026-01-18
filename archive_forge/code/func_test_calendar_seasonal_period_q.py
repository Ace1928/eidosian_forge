from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_calendar_seasonal_period_q():
    period = 'Q'
    index = pd.date_range('2000-01-01', freq=MONTH_END, periods=600)
    cs = CalendarSeasonality(MONTH_END, period=period)
    terms = cs.in_sample(index)
    assert np.all(terms.sum(1) == 1.0)
    for i in range(index.shape[0]):
        assert terms.iloc[i, i % 3] == 1.0