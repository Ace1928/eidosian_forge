from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_pytimedelta(self):
    ser = Series([np.nan, Timedelta('1 days')], index=['A', 'B'])
    result = ser.fillna(timedelta(1))
    expected = Series(Timedelta('1 days'), index=['A', 'B'])
    tm.assert_series_equal(result, expected)