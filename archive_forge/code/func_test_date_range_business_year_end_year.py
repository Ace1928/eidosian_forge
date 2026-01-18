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
def test_date_range_business_year_end_year(self, unit):
    rng = date_range('1/1/2013', '7/1/2017', freq='BYE', unit=unit)
    exp = DatetimeIndex(['2013-12-31', '2014-12-31', '2015-12-31', '2016-12-30'], dtype=f'M8[{unit}]', freq='BYE')
    tm.assert_index_equal(rng, exp)