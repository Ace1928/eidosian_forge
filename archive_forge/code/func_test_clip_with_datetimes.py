from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_clip_with_datetimes(self):
    t = Timestamp('2015-12-01 09:30:30')
    s = Series([Timestamp('2015-12-01 09:30:00'), Timestamp('2015-12-01 09:31:00')])
    result = s.clip(upper=t)
    expected = Series([Timestamp('2015-12-01 09:30:00'), Timestamp('2015-12-01 09:30:30')])
    tm.assert_series_equal(result, expected)
    t = Timestamp('2015-12-01 09:30:30', tz='US/Eastern')
    s = Series([Timestamp('2015-12-01 09:30:00', tz='US/Eastern'), Timestamp('2015-12-01 09:31:00', tz='US/Eastern')])
    result = s.clip(upper=t)
    expected = Series([Timestamp('2015-12-01 09:30:00', tz='US/Eastern'), Timestamp('2015-12-01 09:30:30', tz='US/Eastern')])
    tm.assert_series_equal(result, expected)