from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('dtype', [float, 'float32', 'float64'])
@pytest.mark.parametrize('fill_type', tm.ALL_REAL_NUMPY_DTYPES)
@pytest.mark.parametrize('scalar', [True, False])
def test_fillna_float_casting(self, dtype, fill_type, scalar):
    ser = Series([np.nan, 1.2], dtype=dtype)
    fill_values = Series([2, 2], dtype=fill_type)
    if scalar:
        fill_values = fill_values.dtype.type(2)
    result = ser.fillna(fill_values)
    expected = Series([2.0, 1.2], dtype=dtype)
    tm.assert_series_equal(result, expected)
    ser = Series([np.nan, 1.2], dtype=dtype)
    mask = ser.isna().to_numpy()
    ser[mask] = fill_values
    tm.assert_series_equal(ser, expected)
    ser = Series([np.nan, 1.2], dtype=dtype)
    ser.mask(mask, fill_values, inplace=True)
    tm.assert_series_equal(ser, expected)
    ser = Series([np.nan, 1.2], dtype=dtype)
    res = ser.where(~mask, fill_values)
    tm.assert_series_equal(res, expected)