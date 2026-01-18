import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_min_periods(self, regular):
    df = regular
    expected = df.rolling(2, min_periods=1).sum()
    result = df.rolling('2s').sum()
    tm.assert_frame_equal(result, expected)
    expected = df.rolling(2, min_periods=1).sum()
    result = df.rolling('2s', min_periods=1).sum()
    tm.assert_frame_equal(result, expected)