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
def test_non_iso_strings_with_tz_offset(self):
    result = to_datetime(['March 1, 2018 12:00:00+0400'] * 2)
    expected = DatetimeIndex([datetime(2018, 3, 1, 12, tzinfo=timezone(timedelta(minutes=240)))] * 2)
    tm.assert_index_equal(result, expected)