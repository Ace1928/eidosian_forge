from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.mark.parametrize('dt,exp_week_day,exp_last_day', [(datetime(2017, 11, 30), 3, 30), (datetime(1993, 10, 31), 6, 29)])
def test_get_last_bday(dt, exp_week_day, exp_last_day):
    assert dt.weekday() == exp_week_day
    assert get_lastbday(dt.year, dt.month) == exp_last_day