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
def test_getitem_str_year_with_datetimeindex():
    idx = [Timestamp('2013-05-31 00:00'), Timestamp(datetime(2013, 5, 31, 23, 59, 59, 999999))]
    ts = Series(range(len(idx)), index=idx)
    expected = ts['2013']
    tm.assert_series_equal(expected, ts)