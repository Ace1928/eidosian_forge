import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_timedelta64(self):
    ser = Series([Timedelta('1 days'), Timedelta('2 days'), Timedelta('3 days'), Timedelta('4 days'), Timedelta('5 days')], name='timedelta_data')
    result = ser.describe()
    expected = Series([5, ser[2], ser.std(), ser[0], ser[1], ser[2], ser[3], ser[4]], name='timedelta_data', index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_series_equal(result, expected)