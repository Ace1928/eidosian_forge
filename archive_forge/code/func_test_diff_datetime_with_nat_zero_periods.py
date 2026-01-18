import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_diff_datetime_with_nat_zero_periods(self, tz):
    dti = date_range('2016-01-01', periods=4, tz=tz)
    ser = Series(dti)
    df = ser.to_frame().copy()
    df[1] = ser.copy()
    df.iloc[:, 0] = pd.NaT
    expected = df - df
    assert expected[0].isna().all()
    result = df.diff(0, axis=0)
    tm.assert_frame_equal(result, expected)
    result = df.diff(0, axis=1)
    tm.assert_frame_equal(result, expected)