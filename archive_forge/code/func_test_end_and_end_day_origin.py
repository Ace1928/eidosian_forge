from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('start,end,freq,data,resample_freq,origin,closed,exp_data,exp_end,exp_periods', [('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end', None, [0, 18, 27, 63], '20001002 00:26:00', 4), ('20200101 8:26:35', '20200101 9:31:58', '77s', [1] * 51, '7min', 'end', 'right', [1, 6, 5, 6, 5, 6, 5, 6, 5, 6], '2020-01-01 09:30:45', 10), ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end', 'left', [0, 18, 27, 39, 24], '20001002 00:43:00', 5), ('2000-10-01 23:30:00', '2000-10-02 00:26:00', '7min', [0, 3, 6, 9, 12, 15, 18, 21, 24], '17min', 'end_day', None, [3, 15, 45, 45], '2000-10-02 00:29:00', 4)])
def test_end_and_end_day_origin(start, end, freq, data, resample_freq, origin, closed, exp_data, exp_end, exp_periods):
    rng = date_range(start, end, freq=freq)
    ts = Series(data, index=rng)
    res = ts.resample(resample_freq, origin=origin, closed=closed).sum()
    expected = Series(exp_data, index=date_range(end=exp_end, freq=resample_freq, periods=exp_periods))
    tm.assert_series_equal(res, expected)