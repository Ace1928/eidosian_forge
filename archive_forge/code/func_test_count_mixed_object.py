from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_count_mixed_object():
    ser = Series(['a', np.nan, 'b', True, datetime.today(), 'foo', None, 1, 2.0], dtype=object)
    result = ser.str.count('a')
    expected = Series([1, np.nan, 0, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
    tm.assert_series_equal(result, expected)