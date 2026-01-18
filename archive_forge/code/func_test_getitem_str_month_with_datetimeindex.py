from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_str_month_with_datetimeindex():
    idx = date_range(start='2013-05-31 00:00', end='2013-05-31 23:00', freq='h')
    ts = Series(range(len(idx)), index=idx)
    expected = ts['2013-05']
    tm.assert_series_equal(expected, ts)
    idx = date_range(start='2013-05-31 00:00', end='2013-05-31 23:59', freq='s')
    ts = Series(range(len(idx)), index=idx)
    expected = ts['2013-05']
    tm.assert_series_equal(expected, ts)