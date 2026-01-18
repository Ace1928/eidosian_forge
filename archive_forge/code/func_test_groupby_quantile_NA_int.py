import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_quantile_NA_int(any_int_ea_dtype):
    df = DataFrame({'x': [1, 1], 'y': [2, 5]}, dtype=any_int_ea_dtype)
    result = df.groupby('x')['y'].quantile(0.5)
    expected = pd.Series([3.5], dtype='Float64', index=Index([1], name='x', dtype=any_int_ea_dtype), name='y')
    tm.assert_series_equal(expected, result)
    result = df.groupby('x').quantile(0.5)
    expected = DataFrame({'y': 3.5}, dtype='Float64', index=Index([1], name='x', dtype=any_int_ea_dtype))
    tm.assert_frame_equal(result, expected)