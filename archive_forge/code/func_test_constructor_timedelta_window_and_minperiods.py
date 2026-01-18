from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D'])
def test_constructor_timedelta_window_and_minperiods(window, raw):
    n = 10
    df = DataFrame({'value': np.arange(n)}, index=date_range('2017-08-08', periods=n, freq='D'))
    expected = DataFrame({'value': np.append([np.nan, 1.0], np.arange(3.0, 27.0, 3))}, index=date_range('2017-08-08', periods=n, freq='D'))
    result_roll_sum = df.rolling(window=window, min_periods=2).sum()
    result_roll_generic = df.rolling(window=window, min_periods=2).apply(sum, raw=raw)
    tm.assert_frame_equal(result_roll_sum, expected)
    tm.assert_frame_equal(result_roll_generic, expected)