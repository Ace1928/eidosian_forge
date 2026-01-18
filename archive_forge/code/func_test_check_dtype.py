import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import (
def test_check_dtype(self, data):
    dtype = data.dtype
    df = pd.DataFrame({'A': pd.Series(data, dtype=dtype), 'B': data, 'C': pd.Series(['foo'] * len(data), dtype=object), 'D': 1})
    result = df.dtypes == str(dtype)
    assert np.dtype('int64') != 'Int64'
    expected = pd.Series([True, True, False, False], index=list('ABCD'))
    tm.assert_series_equal(result, expected)
    expected = pd.Series([True, True, False, False], index=list('ABCD'))
    result = df.dtypes.apply(str) == str(dtype)
    tm.assert_series_equal(result, expected)