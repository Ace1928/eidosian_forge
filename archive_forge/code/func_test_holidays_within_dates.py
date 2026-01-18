from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
@pytest.mark.parametrize('holiday,start,expected', [(USMemorialDay, datetime(2015, 7, 1), []), (USMemorialDay, '2015-05-25', [Timestamp('2015-05-25')]), (USLaborDay, datetime(2015, 7, 1), []), (USLaborDay, '2015-09-07', [Timestamp('2015-09-07')]), (USColumbusDay, datetime(2015, 7, 1), []), (USColumbusDay, '2015-10-12', [Timestamp('2015-10-12')]), (USThanksgivingDay, datetime(2015, 7, 1), []), (USThanksgivingDay, '2015-11-26', [Timestamp('2015-11-26')]), (USMartinLutherKingJr, datetime(2015, 7, 1), []), (USMartinLutherKingJr, '2015-01-19', [Timestamp('2015-01-19')]), (USPresidentsDay, datetime(2015, 7, 1), []), (USPresidentsDay, '2015-02-16', [Timestamp('2015-02-16')]), (GoodFriday, datetime(2015, 7, 1), []), (GoodFriday, '2015-04-03', [Timestamp('2015-04-03')]), (EasterMonday, '2015-04-06', [Timestamp('2015-04-06')]), (EasterMonday, datetime(2015, 7, 1), []), (EasterMonday, '2015-04-05', []), ("New Year's Day", '2015-01-01', [Timestamp('2015-01-01')]), ("New Year's Day", '2010-12-31', [Timestamp('2010-12-31')]), ("New Year's Day", datetime(2015, 7, 1), []), ("New Year's Day", '2011-01-01', []), ('Independence Day', '2015-07-03', [Timestamp('2015-07-03')]), ('Independence Day', datetime(2015, 7, 1), []), ('Independence Day', '2015-07-04', []), ('Veterans Day', '2012-11-12', [Timestamp('2012-11-12')]), ('Veterans Day', datetime(2015, 7, 1), []), ('Veterans Day', '2012-11-11', []), ('Christmas Day', '2011-12-26', [Timestamp('2011-12-26')]), ('Christmas Day', datetime(2015, 7, 1), []), ('Christmas Day', '2011-12-25', []), ('Juneteenth National Independence Day', '2020-06-19', []), ('Juneteenth National Independence Day', '2021-06-18', [Timestamp('2021-06-18')]), ('Juneteenth National Independence Day', '2022-06-19', []), ('Juneteenth National Independence Day', '2022-06-20', [Timestamp('2022-06-20')])])
def test_holidays_within_dates(holiday, start, expected):
    if isinstance(holiday, str):
        calendar = get_calendar('USFederalHolidayCalendar')
        holiday = calendar.rule_from_name(holiday)
    assert list(holiday.dates(start, start)) == expected
    assert list(holiday.dates(utc.localize(Timestamp(start)), utc.localize(Timestamp(start)))) == [utc.localize(dt) for dt in expected]