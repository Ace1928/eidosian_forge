from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_numeric_inplace(self):
    x = Series([np.nan, 1.0, np.nan, 3.0, np.nan], ['z', 'a', 'b', 'c', 'd'])
    y = x.copy()
    return_value = y.fillna(value=0, inplace=True)
    assert return_value is None
    expected = x.fillna(value=0)
    tm.assert_series_equal(y, expected)