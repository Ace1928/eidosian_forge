from datetime import timedelta
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dtypes_timedeltas(self):
    df = DataFrame({'A': Series(date_range('2012-1-1', periods=3, freq='D')), 'B': Series([timedelta(days=i) for i in range(3)])})
    result = df.dtypes
    expected = Series([np.dtype('datetime64[ns]'), np.dtype('timedelta64[ns]')], index=list('AB'))
    tm.assert_series_equal(result, expected)
    df['C'] = df['A'] + df['B']
    result = df.dtypes
    expected = Series([np.dtype('datetime64[ns]'), np.dtype('timedelta64[ns]'), np.dtype('datetime64[ns]')], index=list('ABC'))
    tm.assert_series_equal(result, expected)
    df['D'] = 1
    result = df.dtypes
    expected = Series([np.dtype('datetime64[ns]'), np.dtype('timedelta64[ns]'), np.dtype('datetime64[ns]'), np.dtype('int64')], index=list('ABCD'))
    tm.assert_series_equal(result, expected)