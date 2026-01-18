from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_function_returns_numpy_array():

    def fct(group):
        return group['B'].values.flatten()
    df = DataFrame({'A': ['a', 'a', 'b', 'none'], 'B': [1, 2, 3, np.nan]})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('A').apply(fct)
    expected = Series([[1.0, 2.0], [3.0], [np.nan]], index=Index(['a', 'b', 'none'], name='A'))
    tm.assert_series_equal(result, expected)