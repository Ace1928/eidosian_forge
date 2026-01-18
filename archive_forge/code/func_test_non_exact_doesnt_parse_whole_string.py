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
@pytest.mark.parametrize('format, expected', [('%Y-%m-%d', Timestamp(2000, 1, 3)), ('%Y-%d-%m', Timestamp(2000, 3, 1)), ('%Y-%m-%d %H', Timestamp(2000, 1, 3, 12)), ('%Y-%d-%m %H', Timestamp(2000, 3, 1, 12)), ('%Y-%m-%d %H:%M', Timestamp(2000, 1, 3, 12, 34)), ('%Y-%d-%m %H:%M', Timestamp(2000, 3, 1, 12, 34)), ('%Y-%m-%d %H:%M:%S', Timestamp(2000, 1, 3, 12, 34, 56)), ('%Y-%d-%m %H:%M:%S', Timestamp(2000, 3, 1, 12, 34, 56)), ('%Y-%m-%d %H:%M:%S.%f', Timestamp(2000, 1, 3, 12, 34, 56, 123456)), ('%Y-%d-%m %H:%M:%S.%f', Timestamp(2000, 3, 1, 12, 34, 56, 123456)), ('%Y-%m-%d %H:%M:%S.%f%z', Timestamp(2000, 1, 3, 12, 34, 56, 123456, tz='UTC+01:00')), ('%Y-%d-%m %H:%M:%S.%f%z', Timestamp(2000, 3, 1, 12, 34, 56, 123456, tz='UTC+01:00'))])
def test_non_exact_doesnt_parse_whole_string(self, cache, format, expected):
    result = to_datetime('2000-01-03 12:34:56.123456+01:00', format=format, exact=False)
    assert result == expected