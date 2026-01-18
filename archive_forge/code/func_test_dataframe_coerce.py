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
def test_dataframe_coerce(self, cache):
    df2 = DataFrame({'year': [2015, 2016], 'month': [2, 20], 'day': [4, 5]})
    msg = '^cannot assemble the datetimes: time data ".+" doesn\\\'t match format "%Y%m%d", at position 1\\.'
    with pytest.raises(ValueError, match=msg):
        to_datetime(df2, cache=cache)
    result = to_datetime(df2, errors='coerce', cache=cache)
    expected = Series([Timestamp('20150204 00:00:00'), NaT])
    tm.assert_series_equal(result, expected)