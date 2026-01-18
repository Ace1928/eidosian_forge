import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_dropna_period_dtype(self):
    ser = Series([Period('2011-01', freq='M'), Period('NaT', freq='M')])
    result = ser.dropna()
    expected = Series([Period('2011-01', freq='M')])
    tm.assert_series_equal(result, expected)