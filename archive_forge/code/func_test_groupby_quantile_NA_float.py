import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_quantile_NA_float(any_float_dtype):
    df = DataFrame({'x': [1, 1], 'y': [0.2, np.nan]}, dtype=any_float_dtype)
    result = df.groupby('x')['y'].quantile(0.5)
    exp_index = Index([1.0], dtype=any_float_dtype, name='x')
    if any_float_dtype in ['Float32', 'Float64']:
        expected_dtype = any_float_dtype
    else:
        expected_dtype = None
    expected = pd.Series([0.2], dtype=expected_dtype, index=exp_index, name='y')
    tm.assert_series_equal(result, expected)
    result = df.groupby('x')['y'].quantile([0.5, 0.75])
    expected = pd.Series([0.2] * 2, index=pd.MultiIndex.from_product((exp_index, [0.5, 0.75]), names=['x', None]), name='y', dtype=expected_dtype)
    tm.assert_series_equal(result, expected)