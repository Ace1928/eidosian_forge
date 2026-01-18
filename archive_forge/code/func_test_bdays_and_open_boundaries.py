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
@pytest.mark.parametrize('inclusive', ['left', 'right', 'neither', 'both'])
def test_bdays_and_open_boundaries(self, inclusive):
    start = '2018-07-21'
    end = '2018-07-29'
    result = date_range(start, end, freq='B', inclusive=inclusive)
    bday_start = '2018-07-23'
    bday_end = '2018-07-27'
    expected = date_range(bday_start, bday_end, freq='D')
    tm.assert_index_equal(result, expected)