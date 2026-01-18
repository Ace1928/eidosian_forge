import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_with_different_n(self):
    ser = Series([1, 2], index=PeriodIndex(['2020-01', '2020-03'], freq='2M'))
    result = ser.asfreq('M')
    excepted = Series([1, 2], index=PeriodIndex(['2020-02', '2020-04'], freq='M'))
    tm.assert_series_equal(result, excepted)