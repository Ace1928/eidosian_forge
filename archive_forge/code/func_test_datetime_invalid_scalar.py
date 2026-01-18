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
@pytest.mark.parametrize('value', ['a', '00:01:99'])
@pytest.mark.parametrize('format', [None, '%H:%M:%S'])
def test_datetime_invalid_scalar(self, value, format):
    res = to_datetime(value, errors='ignore', format=format)
    assert res == value
    res = to_datetime(value, errors='coerce', format=format)
    assert res is NaT
    msg = '|'.join([f"""^time data "a" doesn\\'t match format "%H:%M:%S", at position 0. {PARSING_ERR_MSG}$""", '^Given date string "a" not likely a datetime, at position 0$', f'^unconverted data remains when parsing with format "%H:%M:%S": "9", at position 0. {PARSING_ERR_MSG}$', '^second must be in 0..59: 00:01:99, at position 0$'])
    with pytest.raises(ValueError, match=msg):
        to_datetime(value, errors='raise', format=format)