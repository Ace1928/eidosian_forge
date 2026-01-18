from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('start, end, freq', [(0.5, None, None), (None, 4.5, None), (0.5, None, 1.5), (None, 6.5, 1.5)])
def test_no_invalid_float_truncation(self, start, end, freq):
    if freq is None:
        breaks = [0.5, 1.5, 2.5, 3.5, 4.5]
    else:
        breaks = [0.5, 2.0, 3.5, 5.0, 6.5]
    expected = IntervalIndex.from_breaks(breaks)
    result = interval_range(start=start, end=end, periods=4, freq=freq)
    tm.assert_index_equal(result, expected)