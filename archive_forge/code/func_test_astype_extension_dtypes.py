import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int64', 'Int32', 'Int16'])
def test_astype_extension_dtypes(self, dtype):
    df = DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=['a', 'b'])
    expected1 = DataFrame({'a': pd.array([1, 3, 5], dtype=dtype), 'b': pd.array([2, 4, 6], dtype=dtype)})
    tm.assert_frame_equal(df.astype(dtype), expected1)
    tm.assert_frame_equal(df.astype('int64').astype(dtype), expected1)
    tm.assert_frame_equal(df.astype(dtype).astype('float64'), df)
    df = DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=['a', 'b'])
    df['b'] = df['b'].astype(dtype)
    expected2 = DataFrame({'a': [1.0, 3.0, 5.0], 'b': pd.array([2, 4, 6], dtype=dtype)})
    tm.assert_frame_equal(df, expected2)
    tm.assert_frame_equal(df.astype(dtype), expected1)
    tm.assert_frame_equal(df.astype('int64').astype(dtype), expected1)