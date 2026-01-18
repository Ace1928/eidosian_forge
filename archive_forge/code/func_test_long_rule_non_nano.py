from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
def test_long_rule_non_nano():
    idx = date_range('0300-01-01', '2000-01-01', unit='s', freq='100YE')
    ser = Series([1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5], index=idx)
    result = ser.resample('200YE').mean()
    expected_idx = DatetimeIndex(np.array(['0300-12-31', '0500-12-31', '0700-12-31', '0900-12-31', '1100-12-31', '1300-12-31', '1500-12-31', '1700-12-31', '1900-12-31']).astype('datetime64[s]'), freq='200YE-DEC')
    expected = Series([1.0, 3.0, 6.5, 4.0, 3.0, 6.5, 4.0, 3.0, 6.5], index=expected_idx)
    tm.assert_series_equal(result, expected)