from datetime import datetime
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_calendar_observance_dates():
    us_fed_cal = get_calendar('USFederalHolidayCalendar')
    holidays0 = us_fed_cal.holidays(datetime(2015, 7, 3), datetime(2015, 7, 3))
    holidays1 = us_fed_cal.holidays(datetime(2015, 7, 3), datetime(2015, 7, 6))
    holidays2 = us_fed_cal.holidays(datetime(2015, 7, 3), datetime(2015, 7, 3))
    tm.assert_index_equal(holidays0, holidays1)
    tm.assert_index_equal(holidays0, holidays2)