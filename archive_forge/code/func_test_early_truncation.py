from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('start, end, freq, expected_endpoint', [(0, 10, 3, 9), (0, 10, 1.5, 9), (0.5, 10, 3, 9.5), (Timedelta('0D'), Timedelta('10D'), '2D4h', Timedelta('8D16h')), (Timestamp('2018-01-01'), Timestamp('2018-02-09'), 'MS', Timestamp('2018-02-01')), (Timestamp('2018-01-01', tz='US/Eastern'), Timestamp('2018-01-20', tz='US/Eastern'), '5D12h', Timestamp('2018-01-17 12:00:00', tz='US/Eastern'))])
def test_early_truncation(self, start, end, freq, expected_endpoint):
    result = interval_range(start=start, end=end, freq=freq)
    result_endpoint = result.right[-1]
    assert result_endpoint == expected_endpoint