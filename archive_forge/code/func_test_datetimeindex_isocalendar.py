import pytest
from pandas import (
import pandas._testing as tm
def test_datetimeindex_isocalendar(using_copy_on_write):
    dt = date_range('2019-12-31', periods=3, freq='D')
    ser = Series(dt)
    df = DatetimeIndex(ser).isocalendar()
    expected = df.index.copy(deep=True)
    ser.iloc[0] = Timestamp('2020-12-31')
    if using_copy_on_write:
        tm.assert_index_equal(df.index, expected)