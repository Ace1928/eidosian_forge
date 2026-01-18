import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_secondly(self):
    ival_S = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=0, second=0)
    ival_S_end_of_year = Period(freq='s', year=2007, month=12, day=31, hour=23, minute=59, second=59)
    ival_S_end_of_quarter = Period(freq='s', year=2007, month=3, day=31, hour=23, minute=59, second=59)
    ival_S_end_of_month = Period(freq='s', year=2007, month=1, day=31, hour=23, minute=59, second=59)
    ival_S_end_of_week = Period(freq='s', year=2007, month=1, day=7, hour=23, minute=59, second=59)
    ival_S_end_of_day = Period(freq='s', year=2007, month=1, day=1, hour=23, minute=59, second=59)
    ival_S_end_of_bus = Period(freq='s', year=2007, month=1, day=1, hour=23, minute=59, second=59)
    ival_S_end_of_hour = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=59, second=59)
    ival_S_end_of_minute = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=0, second=59)
    ival_S_to_A = Period(freq='Y', year=2007)
    ival_S_to_Q = Period(freq='Q', year=2007, quarter=1)
    ival_S_to_M = Period(freq='M', year=2007, month=1)
    ival_S_to_W = Period(freq='W', year=2007, month=1, day=7)
    ival_S_to_D = Period(freq='D', year=2007, month=1, day=1)
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        ival_S_to_B = Period(freq='B', year=2007, month=1, day=1)
    ival_S_to_H = Period(freq='h', year=2007, month=1, day=1, hour=0)
    ival_S_to_T = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0)
    assert ival_S.asfreq('Y') == ival_S_to_A
    assert ival_S_end_of_year.asfreq('Y') == ival_S_to_A
    assert ival_S.asfreq('Q') == ival_S_to_Q
    assert ival_S_end_of_quarter.asfreq('Q') == ival_S_to_Q
    assert ival_S.asfreq('M') == ival_S_to_M
    assert ival_S_end_of_month.asfreq('M') == ival_S_to_M
    assert ival_S.asfreq('W') == ival_S_to_W
    assert ival_S_end_of_week.asfreq('W') == ival_S_to_W
    assert ival_S.asfreq('D') == ival_S_to_D
    assert ival_S_end_of_day.asfreq('D') == ival_S_to_D
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert ival_S.asfreq('B') == ival_S_to_B
        assert ival_S_end_of_bus.asfreq('B') == ival_S_to_B
    assert ival_S.asfreq('h') == ival_S_to_H
    assert ival_S_end_of_hour.asfreq('h') == ival_S_to_H
    assert ival_S.asfreq('Min') == ival_S_to_T
    assert ival_S_end_of_minute.asfreq('Min') == ival_S_to_T
    assert ival_S.asfreq('s') == ival_S