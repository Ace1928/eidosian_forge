from datetime import (
import re
import numpy as np
import pytest
import pytz
from pytz import timezone
from pandas._libs.tslibs import timezones
from pandas._libs.tslibs.offsets import (
from pandas.errors import OutOfBoundsDatetime
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.datetimes import _generate_range as generate_range
from pandas.tests.indexes.datetimes.test_timezones import (
from pandas.tseries.holiday import USFederalHolidayCalendar
def test_date_range_week_of_month2(self, unit):
    result = date_range('2013-1-1', periods=4, freq='WOM-1SAT', unit=unit)
    expected = DatetimeIndex(['2013-01-05', '2013-02-02', '2013-03-02', '2013-04-06'], dtype=f'M8[{unit}]', freq='WOM-1SAT')
    tm.assert_index_equal(result, expected)