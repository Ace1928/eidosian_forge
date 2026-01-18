from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytest
from pandas import Timestamp
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_get_year_end(self):
    assert makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SAT).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 31)
    assert makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.SUN).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 9, 1)
    assert makeFY5253NearestEndMonth(startingMonth=8, weekday=WeekDay.FRI).get_year_end(datetime(2013, 1, 1)) == datetime(2013, 8, 30)
    offset_n = FY5253(weekday=WeekDay.TUE, startingMonth=12, variation='nearest')
    assert offset_n.get_year_end(datetime(2012, 1, 1)) == datetime(2013, 1, 1)
    assert offset_n.get_year_end(datetime(2012, 1, 10)) == datetime(2013, 1, 1)
    assert offset_n.get_year_end(datetime(2013, 1, 1)) == datetime(2013, 12, 31)
    assert offset_n.get_year_end(datetime(2013, 1, 2)) == datetime(2013, 12, 31)
    assert offset_n.get_year_end(datetime(2013, 1, 3)) == datetime(2013, 12, 31)
    assert offset_n.get_year_end(datetime(2013, 1, 10)) == datetime(2013, 12, 31)
    JNJ = FY5253(n=1, startingMonth=12, weekday=6, variation='nearest')
    assert JNJ.get_year_end(datetime(2006, 1, 1)) == datetime(2006, 12, 31)