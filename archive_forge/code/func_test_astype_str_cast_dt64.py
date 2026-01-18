from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_astype_str_cast_dt64(self):
    ts = Series([Timestamp('2010-01-04 00:00:00')])
    res = ts.astype(str)
    expected = Series(['2010-01-04'], dtype=object)
    tm.assert_series_equal(res, expected)
    ts = Series([Timestamp('2010-01-04 00:00:00', tz='US/Eastern')])
    res = ts.astype(str)
    expected = Series(['2010-01-04 00:00:00-05:00'], dtype=object)
    tm.assert_series_equal(res, expected)