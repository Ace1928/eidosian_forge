from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_constructor_with_nonexistent_keyword_arg(self, warsaw):
    timezone = warsaw
    start = Timestamp('2015-03-29 02:30:00').tz_localize(timezone, nonexistent='shift_forward')
    result = date_range(start=start, periods=2, freq='h')
    expected = DatetimeIndex([Timestamp('2015-03-29 03:00:00+02:00', tz=timezone), Timestamp('2015-03-29 04:00:00+02:00', tz=timezone)])
    tm.assert_index_equal(result, expected)
    end = start
    result = date_range(end=end, periods=2, freq='h')
    expected = DatetimeIndex([Timestamp('2015-03-29 01:00:00+01:00', tz=timezone), Timestamp('2015-03-29 03:00:00+02:00', tz=timezone)])
    tm.assert_index_equal(result, expected)