import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'UTC'])
def test_concat_NaT_dataframes(self, tz):
    dti = DatetimeIndex([pd.NaT, pd.NaT], tz=tz)
    first = DataFrame({0: dti})
    second = DataFrame([[Timestamp('2015/01/01', tz=tz)], [Timestamp('2016/01/01', tz=tz)]], index=[2, 3])
    expected = DataFrame([pd.NaT, pd.NaT, Timestamp('2015/01/01', tz=tz), Timestamp('2016/01/01', tz=tz)])
    result = concat([first, second], axis=0)
    tm.assert_frame_equal(result, expected)