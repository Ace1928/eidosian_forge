import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import algorithms
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('values', [[-9.0, 0.0], [-9, 0]])
def test_isin_float_in_int_series(self, values):
    ser = Series(values)
    result = ser.isin([-9, -0.5])
    expected = Series([True, False])
    tm.assert_series_equal(result, expected)