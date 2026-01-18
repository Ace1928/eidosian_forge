from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('closed, window_selections', [('both', [[True, True, False, False, False], [True, True, True, False, False], [False, True, True, True, False], [False, False, True, True, True], [False, False, False, True, True]]), ('left', [[True, False, False, False, False], [True, True, False, False, False], [False, True, True, False, False], [False, False, True, True, False], [False, False, False, True, True]]), ('right', [[True, True, False, False, False], [False, True, True, False, False], [False, False, True, True, False], [False, False, False, True, True], [False, False, False, False, True]]), ('neither', [[True, False, False, False, False], [False, True, False, False, False], [False, False, True, False, False], [False, False, False, True, False], [False, False, False, False, True]])])
def test_datetimelike_centered_selections(closed, window_selections, arithmetic_win_operators):
    func_name = arithmetic_win_operators
    df_time = DataFrame({'A': [0.0, 1.0, 2.0, 3.0, 4.0]}, index=date_range('2020', periods=5))
    expected = DataFrame({'A': [getattr(df_time['A'].iloc[s], func_name)() for s in window_selections]}, index=date_range('2020', periods=5))
    if func_name == 'sem':
        kwargs = {'ddof': 0}
    else:
        kwargs = {}
    result = getattr(df_time.rolling('2D', closed=closed, min_periods=1, center=True), func_name)(**kwargs)
    tm.assert_frame_equal(result, expected, check_dtype=False)