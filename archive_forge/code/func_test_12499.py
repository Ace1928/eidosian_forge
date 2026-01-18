import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason='unwanted casting to dt64')
def test_12499():
    ts = Timestamp('2016-03-01 03:13:22.98986', tz='UTC')
    data = [{'one': 0, 'two': ts}]
    orig = DataFrame(data)
    df = orig.copy()
    df.loc[1] = [np.nan, NaT]
    expected = DataFrame({'one': [0, np.nan], 'two': Series([ts, NaT], dtype='datetime64[ns, UTC]')})
    tm.assert_frame_equal(df, expected)
    data = [{'one': 0, 'two': ts}]
    df = orig.copy()
    df.loc[1, :] = [np.nan, NaT]
    tm.assert_frame_equal(df, expected)