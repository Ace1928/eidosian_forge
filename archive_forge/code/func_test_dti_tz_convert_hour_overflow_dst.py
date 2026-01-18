from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_dti_tz_convert_hour_overflow_dst(self):
    ts = ['2008-05-12 09:50:00', '2008-12-12 09:50:35', '2009-05-12 09:50:32']
    tt = DatetimeIndex(ts).tz_localize('US/Eastern')
    ut = tt.tz_convert('UTC')
    expected = Index([13, 14, 13], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)
    ts = ['2008-05-12 13:50:00', '2008-12-12 14:50:35', '2009-05-12 13:50:32']
    tt = DatetimeIndex(ts).tz_localize('UTC')
    ut = tt.tz_convert('US/Eastern')
    expected = Index([9, 9, 9], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)
    ts = ['2008-05-12 09:50:00', '2008-12-12 09:50:35', '2008-05-12 09:50:32']
    tt = DatetimeIndex(ts).tz_localize('US/Eastern')
    ut = tt.tz_convert('UTC')
    expected = Index([13, 14, 13], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)
    ts = ['2008-05-12 13:50:00', '2008-12-12 14:50:35', '2008-05-12 13:50:32']
    tt = DatetimeIndex(ts).tz_localize('UTC')
    ut = tt.tz_convert('US/Eastern')
    expected = Index([9, 9, 9], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)