import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_describe_ints(self):
    ser = Series([0, 1, 2, 3, 4], name='int_data')
    result = ser.describe()
    expected = Series([5, 2, ser.std(), 0, 1, 2, 3, 4], name='int_data', index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    tm.assert_series_equal(result, expected)