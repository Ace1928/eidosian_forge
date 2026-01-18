import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_diff_datetime_axis0(self, tz):
    df = DataFrame({0: date_range('2010', freq='D', periods=2, tz=tz), 1: date_range('2010', freq='D', periods=2, tz=tz)})
    result = df.diff(axis=0)
    expected = DataFrame({0: pd.TimedeltaIndex(['NaT', '1 days']), 1: pd.TimedeltaIndex(['NaT', '1 days'])})
    tm.assert_frame_equal(result, expected)