from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_fillvalue(self):
    rng = date_range('1/1/2016', periods=10, freq='2s')
    ts = Series(np.arange(len(rng)), index=rng, dtype='float')
    df = DataFrame({'one': ts})
    df.loc['2016-01-01 00:00:08', 'one'] = None
    actual_df = df.asfreq(freq='1s', fill_value=9.0)
    expected_df = df.asfreq(freq='1s').fillna(9.0)
    expected_df.loc['2016-01-01 00:00:08', 'one'] = None
    tm.assert_frame_equal(expected_df, actual_df)
    expected_series = ts.asfreq(freq='1s').fillna(9.0)
    actual_series = ts.asfreq(freq='1s', fill_value=9.0)
    tm.assert_series_equal(expected_series, actual_series)