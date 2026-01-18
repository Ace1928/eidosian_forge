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
@pytest.mark.parametrize('expected, format', [['2015-02-29', None], ['2015-02-29', '%Y-%m-%d'], ['2015-02-29', '%Y-%m-%d'], ['2015-04-31', '%Y-%m-%d']])
def test_day_not_in_month_ignore(self, cache, expected, format):
    result = to_datetime(expected, errors='ignore', format=format, cache=cache)
    assert result == expected