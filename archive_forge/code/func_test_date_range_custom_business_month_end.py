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
def test_date_range_custom_business_month_end(self, unit):
    hcal = USFederalHolidayCalendar()
    freq = offsets.CBMonthEnd(calendar=hcal)
    dti = date_range(start='20120101', end='20130101', freq=freq, unit=unit)
    assert all((freq.is_on_offset(x) for x in dti))
    expected = DatetimeIndex(['2012-01-31', '2012-02-29', '2012-03-30', '2012-04-30', '2012-05-31', '2012-06-29', '2012-07-31', '2012-08-31', '2012-09-28', '2012-10-31', '2012-11-30', '2012-12-31'], dtype=f'M8[{unit}]', freq=freq)
    tm.assert_index_equal(dti, expected)