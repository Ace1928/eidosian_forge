import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_all2(self, arithmetic_win_operators):
    f = arithmetic_win_operators
    df = DataFrame({'B': np.arange(50)}, index=date_range('20130101', periods=50, freq='h'))
    dft = df.between_time('09:00', '16:00')
    r = dft.rolling(window='5h')
    result = getattr(r, f)()

    def agg_by_day(x):
        x = x.between_time('09:00', '16:00')
        return getattr(x.rolling(5, min_periods=1), f)()
    expected = df.groupby(df.index.day).apply(agg_by_day).reset_index(level=0, drop=True)
    tm.assert_frame_equal(result, expected)