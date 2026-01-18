import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_quantile_dt64tz_period():
    dti = pd.date_range('2016-01-01', periods=1000)
    ser = pd.Series(dti)
    df = ser.to_frame()
    df[1] = dti.tz_localize('US/Pacific')
    df[2] = dti.to_period('D')
    df[3] = dti - dti[0]
    df.iloc[-1] = pd.NaT
    by = np.tile(np.arange(5), 200)
    gb = df.groupby(by)
    result = gb.quantile(0.5)
    exp = {i: df.iloc[i::5].quantile(0.5) for i in range(5)}
    expected = DataFrame(exp).T
    expected.index = expected.index.astype(np.int_)
    tm.assert_frame_equal(result, expected)