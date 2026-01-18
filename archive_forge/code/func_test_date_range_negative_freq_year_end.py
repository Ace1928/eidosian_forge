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
def test_date_range_negative_freq_year_end(self, unit):
    rng = date_range('2011-12-31', freq='-2YE', periods=3, unit=unit)
    exp = DatetimeIndex(['2011-12-31', '2009-12-31', '2007-12-31'], dtype=f'M8[{unit}]', freq='-2YE')
    tm.assert_index_equal(rng, exp)
    assert rng.freq == '-2YE'