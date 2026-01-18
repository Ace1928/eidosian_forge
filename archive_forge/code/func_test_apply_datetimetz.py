import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_datetimetz(by_row):
    values = date_range('2011-01-01', '2011-01-02', freq='h').tz_localize('Asia/Tokyo')
    s = Series(values, name='XX')
    result = s.apply(lambda x: x + pd.offsets.Day(), by_row=by_row)
    exp_values = date_range('2011-01-02', '2011-01-03', freq='h').tz_localize('Asia/Tokyo')
    exp = Series(exp_values, name='XX')
    tm.assert_series_equal(result, exp)
    result = s.apply(lambda x: x.hour if by_row else x.dt.hour, by_row=by_row)
    exp = Series(list(range(24)) + [0], name='XX', dtype='int64' if by_row else 'int32')
    tm.assert_series_equal(result, exp)

    def f(x):
        return str(x.tz) if by_row else str(x.dt.tz)
    result = s.apply(f, by_row=by_row)
    if by_row:
        exp = Series(['Asia/Tokyo'] * 25, name='XX')
        tm.assert_series_equal(result, exp)
    else:
        assert result == 'Asia/Tokyo'