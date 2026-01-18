import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_argsort_dt64(self, unit):
    ser = Series([Timestamp(f'201301{i:02d}') for i in range(1, 6)], dtype=f'M8[{unit}]')
    assert ser.dtype == f'datetime64[{unit}]'
    shifted = ser.shift(-1)
    assert shifted.dtype == f'datetime64[{unit}]'
    assert isna(shifted[4])
    result = ser.argsort()
    expected = Series(range(5), dtype=np.intp)
    tm.assert_series_equal(result, expected)
    msg = 'The behavior of Series.argsort in the presence of NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = shifted.argsort()
    expected = Series(list(range(4)) + [-1], dtype=np.intp)
    tm.assert_series_equal(result, expected)