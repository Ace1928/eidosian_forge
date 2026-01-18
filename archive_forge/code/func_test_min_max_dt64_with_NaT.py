from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_min_max_dt64_with_NaT(self):
    df = DataFrame({'foo': [pd.NaT, pd.NaT, Timestamp('2012-05-01')]})
    res = df.min()
    exp = Series([Timestamp('2012-05-01')], index=['foo'])
    tm.assert_series_equal(res, exp)
    res = df.max()
    exp = Series([Timestamp('2012-05-01')], index=['foo'])
    tm.assert_series_equal(res, exp)
    df = DataFrame({'foo': [pd.NaT, pd.NaT]})
    res = df.min()
    exp = Series([pd.NaT], index=['foo'])
    tm.assert_series_equal(res, exp)
    res = df.max()
    exp = Series([pd.NaT], index=['foo'])
    tm.assert_series_equal(res, exp)