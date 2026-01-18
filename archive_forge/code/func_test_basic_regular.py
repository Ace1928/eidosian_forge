import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_basic_regular(self, regular):
    df = regular.copy()
    df.index = date_range('20130101', periods=5, freq='D')
    expected = df.rolling(window=1, min_periods=1).sum()
    result = df.rolling(window='1D').sum()
    tm.assert_frame_equal(result, expected)
    df.index = date_range('20130101', periods=5, freq='2D')
    expected = df.rolling(window=1, min_periods=1).sum()
    result = df.rolling(window='2D', min_periods=1).sum()
    tm.assert_frame_equal(result, expected)
    expected = df.rolling(window=1, min_periods=1).sum()
    result = df.rolling(window='2D', min_periods=1).sum()
    tm.assert_frame_equal(result, expected)
    expected = df.rolling(window=1).sum()
    result = df.rolling(window='2D').sum()
    tm.assert_frame_equal(result, expected)