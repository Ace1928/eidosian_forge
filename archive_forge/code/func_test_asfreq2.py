from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq2(self, frame_or_series):
    ts = frame_or_series([0.0, 1.0, 2.0], index=DatetimeIndex([datetime(2009, 10, 30), datetime(2009, 11, 30), datetime(2009, 12, 31)], dtype='M8[ns]', freq='BME'))
    daily_ts = ts.asfreq('B')
    monthly_ts = daily_ts.asfreq('BME')
    tm.assert_equal(monthly_ts, ts)
    daily_ts = ts.asfreq('B', method='pad')
    monthly_ts = daily_ts.asfreq('BME')
    tm.assert_equal(monthly_ts, ts)
    daily_ts = ts.asfreq(offsets.BDay())
    monthly_ts = daily_ts.asfreq(offsets.BMonthEnd())
    tm.assert_equal(monthly_ts, ts)
    result = ts[:0].asfreq('ME')
    assert len(result) == 0
    assert result is not ts
    if frame_or_series is Series:
        daily_ts = ts.asfreq('D', fill_value=-1)
        result = daily_ts.value_counts().sort_index()
        expected = Series([60, 1, 1, 1], index=[-1.0, 2.0, 1.0, 0.0], name='count').sort_index()
        tm.assert_series_equal(result, expected)