from datetime import datetime
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_no_holidays_calendar():

    class NoHolidaysCalendar(AbstractHolidayCalendar):
        pass
    cal = NoHolidaysCalendar()
    holidays = cal.holidays(Timestamp('01-Jan-2020'), Timestamp('01-Jan-2021'))
    empty_index = DatetimeIndex([])
    tm.assert_index_equal(holidays, empty_index)