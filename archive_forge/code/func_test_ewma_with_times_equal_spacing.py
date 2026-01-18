import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('times', [np.arange(10).astype('datetime64[D]').astype('datetime64[ns]'), date_range('2000', freq='D', periods=10), date_range('2000', freq='D', periods=10).tz_localize('UTC')])
@pytest.mark.parametrize('min_periods', [0, 2])
def test_ewma_with_times_equal_spacing(halflife_with_times, times, min_periods):
    halflife = halflife_with_times
    data = np.arange(10.0)
    data[::2] = np.nan
    df = DataFrame({'A': data})
    result = df.ewm(halflife=halflife, min_periods=min_periods, times=times).mean()
    expected = df.ewm(halflife=1.0, min_periods=min_periods).mean()
    tm.assert_frame_equal(result, expected)