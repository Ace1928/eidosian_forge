from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_check_index_type():
    ct = CalendarTimeTrend(YEAR_END, True, order=3)
    idx = pd.RangeIndex(0, 20)
    with pytest.raises(TypeError, match='CalendarTimeTrend terms can only'):
        ct._check_index_type(idx, pd.DatetimeIndex)
    with pytest.raises(TypeError, match='CalendarTimeTrend terms can only'):
        ct._check_index_type(idx, (pd.DatetimeIndex,))
    with pytest.raises(TypeError, match='CalendarTimeTrend terms can only'):
        ct._check_index_type(idx, (pd.DatetimeIndex, pd.PeriodIndex))
    idx = pd.Index([0, 1, 1, 2, 3, 5, 8, 13])
    with pytest.raises(TypeError, match='CalendarTimeTrend terms can only'):
        types = (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)
        ct._check_index_type(idx, types)