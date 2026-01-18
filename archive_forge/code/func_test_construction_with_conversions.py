from datetime import (
import itertools
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
def test_construction_with_conversions(self):
    arr = np.array([1, 2, 3], dtype='timedelta64[s]')
    df = DataFrame(index=range(3))
    df['A'] = arr
    expected = DataFrame({'A': pd.timedelta_range('00:00:01', periods=3, freq='s')}, index=range(3))
    tm.assert_numpy_array_equal(df['A'].to_numpy(), arr)
    expected = DataFrame({'dt1': Timestamp('20130101'), 'dt2': date_range('20130101', periods=3).astype('M8[s]')}, index=range(3))
    assert expected.dtypes['dt1'] == 'M8[s]'
    assert expected.dtypes['dt2'] == 'M8[s]'
    df = DataFrame(index=range(3))
    df['dt1'] = np.datetime64('2013-01-01')
    df['dt2'] = np.array(['2013-01-01', '2013-01-02', '2013-01-03'], dtype='datetime64[D]')
    tm.assert_frame_equal(df, expected)