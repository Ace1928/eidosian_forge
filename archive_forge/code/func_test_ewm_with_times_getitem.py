import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_with_times_getitem(halflife_with_times):
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    times = date_range('2000', freq='D', periods=10)
    df = DataFrame({'A': data, 'B': data})
    result = df.ewm(halflife=halflife, times=times)['A'].mean()
    expected = df.ewm(halflife=1.0)['A'].mean()
    tm.assert_series_equal(result, expected)