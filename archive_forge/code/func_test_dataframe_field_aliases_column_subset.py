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
@pytest.mark.parametrize('unit', [{'year': 'years', 'month': 'months', 'day': 'days', 'hour': 'hours', 'minute': 'minutes', 'second': 'seconds'}, {'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour', 'minute': 'minute', 'second': 'second'}])
def test_dataframe_field_aliases_column_subset(self, df, cache, unit):
    result = to_datetime(df[list(unit.keys())].rename(columns=unit), cache=cache)
    expected = Series([Timestamp('20150204 06:58:10'), Timestamp('20160305 07:59:11')], dtype='M8[ns]')
    tm.assert_series_equal(result, expected)