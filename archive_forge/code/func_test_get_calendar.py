from datetime import datetime
import pytest
from pytz import utc
from pandas import (
import pandas._testing as tm
from pandas.tseries.holiday import (
def test_get_calendar():

    class TestCalendar(AbstractHolidayCalendar):
        rules = []
    calendar = get_calendar('TestCalendar')
    assert TestCalendar == type(calendar)