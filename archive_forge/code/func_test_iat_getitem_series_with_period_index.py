import numpy as np
from pandas import (
import pandas._testing as tm
def test_iat_getitem_series_with_period_index():
    index = period_range('1/1/2001', periods=10)
    ser = Series(np.random.default_rng(2).standard_normal(10), index=index)
    expected = ser[index[0]]
    result = ser.iat[0]
    assert expected == result