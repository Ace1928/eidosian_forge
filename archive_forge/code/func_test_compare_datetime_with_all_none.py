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
def test_compare_datetime_with_all_none():
    ser = Series(['2020-01-01', '2020-01-02'], dtype='datetime64[ns]')
    ser2 = Series([None, None])
    result = ser > ser2
    expected = Series([False, False])
    tm.assert_series_equal(result, expected)