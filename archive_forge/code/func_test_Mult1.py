from datetime import datetime
from dateutil.tz.tz import tzlocal
import pytest
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('offset_box, offset1', [[BDay, BDay()], [LastWeekOfMonth, LastWeekOfMonth()], [WeekOfMonth, WeekOfMonth()], [Week, Week()], [SemiMonthBegin, SemiMonthBegin()], [SemiMonthEnd, SemiMonthEnd()], [CustomBusinessHour, CustomBusinessHour(weekmask='Tue Wed Thu Fri')], [BusinessHour, BusinessHour()]])
def test_Mult1(offset_box, offset1):
    dt = Timestamp(2008, 1, 2)
    assert dt + 10 * offset1 == dt + offset_box(10)
    assert dt + 5 * offset1 == dt + offset_box(5)