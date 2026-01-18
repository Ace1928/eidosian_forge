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
@pytest.mark.parametrize('arg, format, msg', [('2015-02-29', '%Y-%m-%d', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$'), ('2015-29-02', '%Y-%d-%m', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$'), ('2015-02-32', '%Y-%m-%d', f'^unconverted data remains when parsing with format "%Y-%m-%d": "2", at position 0. {PARSING_ERR_MSG}$'), ('2015-32-02', '%Y-%d-%m', f"""^time data "2015-32-02" doesn't match format "%Y-%d-%m", at position 0. {PARSING_ERR_MSG}$"""), ('2015-04-31', '%Y-%m-%d', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$'), ('2015-31-04', '%Y-%d-%m', f'^day is out of range for month, at position 0. {PARSING_ERR_MSG}$')])
def test_day_not_in_month_raise_value(self, cache, arg, format, msg):
    with pytest.raises(ValueError, match=msg):
        to_datetime(arg, errors='raise', format=format, cache=cache)