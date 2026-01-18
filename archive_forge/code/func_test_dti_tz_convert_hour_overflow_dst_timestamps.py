from datetime import datetime
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_tz_convert_hour_overflow_dst_timestamps(self, tz):
    ts = [Timestamp('2008-05-12 09:50:00', tz=tz), Timestamp('2008-12-12 09:50:35', tz=tz), Timestamp('2009-05-12 09:50:32', tz=tz)]
    tt = DatetimeIndex(ts)
    ut = tt.tz_convert('UTC')
    expected = Index([13, 14, 13], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)
    ts = [Timestamp('2008-05-12 13:50:00', tz='UTC'), Timestamp('2008-12-12 14:50:35', tz='UTC'), Timestamp('2009-05-12 13:50:32', tz='UTC')]
    tt = DatetimeIndex(ts)
    ut = tt.tz_convert('US/Eastern')
    expected = Index([9, 9, 9], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)
    ts = [Timestamp('2008-05-12 09:50:00', tz=tz), Timestamp('2008-12-12 09:50:35', tz=tz), Timestamp('2008-05-12 09:50:32', tz=tz)]
    tt = DatetimeIndex(ts)
    ut = tt.tz_convert('UTC')
    expected = Index([13, 14, 13], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)
    ts = [Timestamp('2008-05-12 13:50:00', tz='UTC'), Timestamp('2008-12-12 14:50:35', tz='UTC'), Timestamp('2008-05-12 13:50:32', tz='UTC')]
    tt = DatetimeIndex(ts)
    ut = tt.tz_convert('US/Eastern')
    expected = Index([9, 9, 9], dtype=np.int32)
    tm.assert_index_equal(ut.hour, expected)