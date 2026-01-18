import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_period(self):
    ser = Series([Period('2020-01', 'M'), Period('2020-01', 'M'), Period('2019-12', 'M')], name='period_data')
    result = ser.describe()
    expected = Series([3, 2, ser[0], 2], name='period_data', index=['count', 'unique', 'top', 'freq'])
    tm.assert_series_equal(result, expected)