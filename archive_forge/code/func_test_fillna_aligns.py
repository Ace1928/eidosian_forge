from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_fillna_aligns(self):
    s1 = Series([0, 1, 2], list('abc'))
    s2 = Series([0, np.nan, 2], list('bac'))
    result = s2.fillna(s1)
    expected = Series([0, 0, 2.0], list('bac'))
    tm.assert_series_equal(result, expected)