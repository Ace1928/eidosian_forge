import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_diff_datetime_axis0_with_nat(self, tz, unit):
    dti = pd.DatetimeIndex(['NaT', '2019-01-01', '2019-01-02'], tz=tz).as_unit(unit)
    ser = Series(dti)
    df = ser.to_frame()
    result = df.diff()
    ex_index = pd.TimedeltaIndex([pd.NaT, pd.NaT, pd.Timedelta(days=1)]).as_unit(unit)
    expected = Series(ex_index).to_frame()
    tm.assert_frame_equal(result, expected)