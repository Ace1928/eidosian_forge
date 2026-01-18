import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype, expected_data, expected_index, axis, expected_dtype', [['datetime64[ns]', [], [], 1, 'datetime64[ns]'], ['datetime64[ns]', [pd.NaT, pd.NaT], ['a', 'b'], 0, 'datetime64[ns]']])
def test_empty_datelike(self, dtype, expected_data, expected_index, axis, expected_dtype):
    df = DataFrame(columns=['a', 'b'], dtype=dtype)
    result = df.quantile(0.5, axis=axis, numeric_only=False)
    expected = Series(expected_data, name=0.5, index=Index(expected_index), dtype=expected_dtype)
    tm.assert_series_equal(result, expected)