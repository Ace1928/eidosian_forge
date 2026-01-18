import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_left_index_right_index_tolerance(self, unit):
    if unit == 's':
        pytest.skip("This test is invalid with unit='s' because that would round dr1")
    dr1 = pd.date_range(start='1/1/2020', end='1/20/2020', freq='2D', unit=unit) + Timedelta(seconds=0.4).as_unit(unit)
    dr2 = pd.date_range(start='1/1/2020', end='2/1/2020', unit=unit)
    df1 = pd.DataFrame({'val1': 'foo'}, index=pd.DatetimeIndex(dr1))
    df2 = pd.DataFrame({'val2': 'bar'}, index=pd.DatetimeIndex(dr2))
    expected = pd.DataFrame({'val1': 'foo', 'val2': 'bar'}, index=pd.DatetimeIndex(dr1))
    result = merge_asof(df1, df2, left_index=True, right_index=True, tolerance=Timedelta(seconds=0.5))
    tm.assert_frame_equal(result, expected)