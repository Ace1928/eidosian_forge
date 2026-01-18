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
def test_date_range_timedelta(self):
    start = '2020-01-01'
    end = '2020-01-11'
    rng1 = date_range(start, end, freq='3D')
    rng2 = date_range(start, end, freq=timedelta(days=3))
    tm.assert_index_equal(rng1, rng2)