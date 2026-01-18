import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_bools(self):
    ser = Series([True, True, False, False, False], name='bool_data')
    result = ser.describe()
    expected = Series([5, 2, False, 3], name='bool_data', index=['count', 'unique', 'top', 'freq'])
    tm.assert_series_equal(result, expected)