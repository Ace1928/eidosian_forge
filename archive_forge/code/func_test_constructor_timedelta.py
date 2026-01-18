from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('freq, periods', [('D', 100), ('2D12h', 40), ('5D', 20), ('25D', 4)])
def test_constructor_timedelta(self, closed, name, freq, periods):
    start, end = (Timedelta('0 days'), Timedelta('100 days'))
    breaks = timedelta_range(start=start, end=end, freq=freq)
    expected = IntervalIndex.from_breaks(breaks, name=name, closed=closed)
    result = interval_range(start=start, end=end, freq=freq, name=name, closed=closed)
    tm.assert_index_equal(result, expected)
    result = interval_range(start=start, periods=periods, freq=freq, name=name, closed=closed)
    tm.assert_index_equal(result, expected)
    result = interval_range(end=end, periods=periods, freq=freq, name=name, closed=closed)
    tm.assert_index_equal(result, expected)
    result = interval_range(start=start, end=end, periods=periods, name=name, closed=closed)
    tm.assert_index_equal(result, expected)