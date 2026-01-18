import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_box_dt64tz():
    vals = [pd.Timestamp('2011-01-01', tz='US/Eastern'), pd.Timestamp('2011-01-02', tz='US/Eastern')]
    ser = Series(vals, dtype='M8[ns, US/Eastern]')
    assert ser.dtype == 'datetime64[ns, US/Eastern]'
    res = ser.apply(lambda x: f'{type(x).__name__}_{x.day}_{x.tz}', by_row='compat')
    exp = Series(['Timestamp_1_US/Eastern', 'Timestamp_2_US/Eastern'])
    tm.assert_series_equal(res, exp)