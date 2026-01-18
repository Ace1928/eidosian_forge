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
def test_date_range_fractional_period(self):
    msg = "Non-integer 'periods' in pd.date_range, pd.timedelta_range"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rng = date_range('1/1/2000', periods=10.5)
    exp = date_range('1/1/2000', periods=10)
    tm.assert_index_equal(rng, exp)