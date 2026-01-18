import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_datetime_columns(self):
    columns = pd.DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01'], freq='MS', tz='US/Eastern', name='XXX')
    df = DataFrame({0: [10, 20, 30, 40, 50], 1: [10, 20, 30, 40, 50], 2: ['A', 0, None, 'X', 1]})
    df.columns = columns
    result = df.describe()
    exp_columns = pd.DatetimeIndex(['2011-01-01', '2011-02-01'], freq='MS', tz='US/Eastern', name='XXX')
    expected = DataFrame({0: [5, 30, df.iloc[:, 0].std(), 10, 20, 30, 40, 50], 1: [5, 30, df.iloc[:, 1].std(), 10, 20, 30, 40, 50]}, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    expected.columns = exp_columns
    tm.assert_frame_equal(result, expected)
    assert result.columns.freq == 'MS'
    assert result.columns.tz == expected.columns.tz