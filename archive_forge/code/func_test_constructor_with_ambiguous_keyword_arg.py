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
def test_constructor_with_ambiguous_keyword_arg(self):
    expected = DatetimeIndex(['2020-11-01 01:00:00', '2020-11-02 01:00:00'], dtype='datetime64[ns, America/New_York]', freq='D', ambiguous=False)
    timezone = 'America/New_York'
    start = Timestamp(year=2020, month=11, day=1, hour=1).tz_localize(timezone, ambiguous=False)
    result = date_range(start=start, periods=2, ambiguous=False)
    tm.assert_index_equal(result, expected)
    timezone = 'America/New_York'
    end = Timestamp(year=2020, month=11, day=2, hour=1).tz_localize(timezone, ambiguous=False)
    result = date_range(end=end, periods=2, ambiguous=False)
    tm.assert_index_equal(result, expected)