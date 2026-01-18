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
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_date_range_timezone_str_argument(self, tzstr):
    tz = timezones.maybe_get_tz(tzstr)
    result = date_range('1/1/2000', periods=10, tz=tzstr)
    expected = date_range('1/1/2000', periods=10, tz=tz)
    tm.assert_index_equal(result, expected)