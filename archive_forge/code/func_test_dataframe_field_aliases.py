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
def test_dataframe_field_aliases(self, df, cache):
    d = {'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour', 'minute': 'minute', 'second': 'second', 'ms': 'ms', 'us': 'us', 'ns': 'ns'}
    result = to_datetime(df.rename(columns=d), cache=cache)
    expected = Series([Timestamp('20150204 06:58:10.001002003'), Timestamp('20160305 07:59:11.001002003')])
    tm.assert_series_equal(result, expected)