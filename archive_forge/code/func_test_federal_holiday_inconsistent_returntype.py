from datetime import datetime
from pandas import DatetimeIndex
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_federal_holiday_inconsistent_returntype():
    cal1 = USFederalHolidayCalendar()
    cal2 = USFederalHolidayCalendar()
    results_2018 = cal1.holidays(start=datetime(2018, 8, 1), end=datetime(2018, 8, 31))
    results_2019 = cal2.holidays(start=datetime(2019, 8, 1), end=datetime(2019, 8, 31))
    expected_results = DatetimeIndex([], dtype='datetime64[ns]', freq=None)
    tm.assert_index_equal(results_2018, expected_results)
    tm.assert_index_equal(results_2019, expected_results)