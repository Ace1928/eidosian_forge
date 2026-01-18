import pytest
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import OutOfBoundsDatetime
from pandas import (
import pandas._testing as tm
def test_conv_monthly(self):
    ival_M = Period(freq='M', year=2007, month=1)
    ival_M_end_of_year = Period(freq='M', year=2007, month=12)
    ival_M_end_of_quarter = Period(freq='M', year=2007, month=3)
    ival_M_to_A = Period(freq='Y', year=2007)
    ival_M_to_Q = Period(freq='Q', year=2007, quarter=1)
    ival_M_to_W_start = Period(freq='W', year=2007, month=1, day=1)
    ival_M_to_W_end = Period(freq='W', year=2007, month=1, day=31)
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        ival_M_to_B_start = Period(freq='B', year=2007, month=1, day=1)
        ival_M_to_B_end = Period(freq='B', year=2007, month=1, day=31)
    ival_M_to_D_start = Period(freq='D', year=2007, month=1, day=1)
    ival_M_to_D_end = Period(freq='D', year=2007, month=1, day=31)
    ival_M_to_H_start = Period(freq='h', year=2007, month=1, day=1, hour=0)
    ival_M_to_H_end = Period(freq='h', year=2007, month=1, day=31, hour=23)
    ival_M_to_T_start = Period(freq='Min', year=2007, month=1, day=1, hour=0, minute=0)
    ival_M_to_T_end = Period(freq='Min', year=2007, month=1, day=31, hour=23, minute=59)
    ival_M_to_S_start = Period(freq='s', year=2007, month=1, day=1, hour=0, minute=0, second=0)
    ival_M_to_S_end = Period(freq='s', year=2007, month=1, day=31, hour=23, minute=59, second=59)
    assert ival_M.asfreq('Y') == ival_M_to_A
    assert ival_M_end_of_year.asfreq('Y') == ival_M_to_A
    assert ival_M.asfreq('Q') == ival_M_to_Q
    assert ival_M_end_of_quarter.asfreq('Q') == ival_M_to_Q
    assert ival_M.asfreq('W', 's') == ival_M_to_W_start
    assert ival_M.asfreq('W', 'E') == ival_M_to_W_end
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        assert ival_M.asfreq('B', 's') == ival_M_to_B_start
        assert ival_M.asfreq('B', 'E') == ival_M_to_B_end
    assert ival_M.asfreq('D', 's') == ival_M_to_D_start
    assert ival_M.asfreq('D', 'E') == ival_M_to_D_end
    assert ival_M.asfreq('h', 's') == ival_M_to_H_start
    assert ival_M.asfreq('h', 'E') == ival_M_to_H_end
    assert ival_M.asfreq('Min', 's') == ival_M_to_T_start
    assert ival_M.asfreq('Min', 'E') == ival_M_to_T_end
    assert ival_M.asfreq('s', 's') == ival_M_to_S_start
    assert ival_M.asfreq('s', 'E') == ival_M_to_S_end
    assert ival_M.asfreq('M') == ival_M