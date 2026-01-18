from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_holidays_with_timezone_specified_but_no_occurences():
    start_date = Timestamp('2018-01-01', tz='America/Chicago')
    end_date = Timestamp('2018-01-11', tz='America/Chicago')
    test_case = USFederalHolidayCalendar().holidays(start_date, end_date, return_name=True)
    expected_results = Series("New Year's Day", index=[start_date])
    expected_results.index = expected_results.index.as_unit('ns')
    tm.assert_equal(test_case, expected_results)