import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_hourly(self):
    ival_H = Period(freq='h', year=2007, month=1, day=1, hour=0)
    ival_H_end_of_year = Period(freq='h', year=2007, month=12, day=31, hour=23)
    ival_H_end_of_quarter = Period(freq='h', year=2007, month=3, day=31, hour=23)
    ival_H_end_of_month = Period(freq='h', year=2007, month=1, day=31, hour=23)
    ival_H_end_of_week = Period(freq='h', year=2007, month=1, day=7, hour=23)
    ival_H_end_of_day = Period(freq='h', year=2007, month=1, day=1, hour=23)
    ival_H_end_of_bus = Period(freq='h', year=2007, month=1, day=1, hour=23)
    ival_H_to_A = Period(freq='Y', year=2007)
    ival_H_to_Q = Period(freq='Q', year=2007, quarter=1)
    ival_H_to_M = Period(freq='M', year=2007, month=1)
    ival_H_to_W = Period(freq='W', year=2007, month=1, day=7)
    ival_H_to_D = Period(freq='D', year=2007, month=1, day=1)
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        ival_H_to_B = Period(freq='B', year=2007, month=1, day=1)
    ival_H_to_T_start = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0)
    ival_H_to_T_end = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=59)
    ival_H_to_S_start = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=0, second=0)
    ival_H_to_S_end = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=59, second=59)
    assert ival_H.asfreq('Y') == ival_H_to_A
    assert ival_H_end_of_year.asfreq('Y') == ival_H_to_A
    assert ival_H.asfreq('Q') == ival_H_to_Q
    assert ival_H_end_of_quarter.asfreq('Q') == ival_H_to_Q
    assert ival_H.asfreq('M') == ival_H_to_M
    assert ival_H_end_of_month.asfreq('M') == ival_H_to_M
    assert ival_H.asfreq('W') == ival_H_to_W
    assert ival_H_end_of_week.asfreq('W') == ival_H_to_W
    assert ival_H.asfreq('D') == ival_H_to_D
    assert ival_H_end_of_day.asfreq('D') == ival_H_to_D
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert ival_H.asfreq('B') == ival_H_to_B
        assert ival_H_end_of_bus.asfreq('B') == ival_H_to_B
    assert ival_H.asfreq('Min', 's') == ival_H_to_T_start
    assert ival_H.asfreq('Min', 'E') == ival_H_to_T_end
    assert ival_H.asfreq('s', 's') == ival_H_to_S_start
    assert ival_H.asfreq('s', 'E') == ival_H_to_S_end
    assert ival_H.asfreq('h') == ival_H