import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
def test_iso_8601_strings_with_same_offset(self):
    ts_str = '2015-11-18 15:30:00+05:30'
    result = to_datetime(ts_str)
    expected = Timestamp(ts_str)
    assert result == expected
    expected = DatetimeIndex([Timestamp(ts_str)] * 2)
    result = to_datetime([ts_str] * 2)
    tm.assert_index_equal(result, expected)
    result = DatetimeIndex([ts_str] * 2)
    tm.assert_index_equal(result, expected)