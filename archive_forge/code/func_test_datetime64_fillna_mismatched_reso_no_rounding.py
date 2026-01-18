from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('scalar', [False, pytest.param(True, marks=pytest.mark.xfail(reason='GH#56410 scalar case not yet addressed'))])
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_datetime64_fillna_mismatched_reso_no_rounding(self, tz, scalar):
    dti = date_range('2016-01-01', periods=3, unit='s', tz=tz)
    item = Timestamp('2016-02-03 04:05:06.789', tz=tz)
    vec = date_range(item, periods=3, unit='ms')
    exp_dtype = 'M8[ms]' if tz is None else 'M8[ms, UTC]'
    expected = Series([item, dti[1], dti[2]], dtype=exp_dtype)
    ser = Series(dti)
    ser[0] = NaT
    ser2 = ser.copy()
    res = ser.fillna(item)
    res2 = ser2.fillna(Series(vec))
    if scalar:
        tm.assert_series_equal(res, expected)
    else:
        tm.assert_series_equal(res2, expected)