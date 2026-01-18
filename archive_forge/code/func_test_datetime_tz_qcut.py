import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('bins', [3, np.linspace(0, 1, 4)])
def test_datetime_tz_qcut(bins):
    tz = 'US/Eastern'
    ser = Series(date_range('20130101', periods=3, tz=tz))
    result = qcut(ser, bins)
    expected = Series(IntervalIndex([Interval(Timestamp('2012-12-31 23:59:59.999999999', tz=tz), Timestamp('2013-01-01 16:00:00', tz=tz)), Interval(Timestamp('2013-01-01 16:00:00', tz=tz), Timestamp('2013-01-02 08:00:00', tz=tz)), Interval(Timestamp('2013-01-02 08:00:00', tz=tz), Timestamp('2013-01-03 00:00:00', tz=tz))])).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)