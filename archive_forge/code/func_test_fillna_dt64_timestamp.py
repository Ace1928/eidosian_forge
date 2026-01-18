from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_dt64_timestamp(self, frame_or_series):
    ser = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130102'), Timestamp('20130103 9:01:01')])
    ser[2] = np.nan
    obj = frame_or_series(ser)
    result = obj.fillna(Timestamp('20130104'))
    expected = Series([Timestamp('20130101'), Timestamp('20130101'), Timestamp('20130104'), Timestamp('20130103 9:01:01')])
    expected = frame_or_series(expected)
    tm.assert_equal(result, expected)
    result = obj.fillna(NaT)
    expected = obj
    tm.assert_equal(result, expected)