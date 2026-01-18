import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('axis,expected', [(0, DataFrame({'a': [np.nan, 0, 1, 0, np.nan, np.nan, np.nan, 0], 'b': [np.nan, 1, np.nan, np.nan, -2, 1, np.nan, np.nan], 'c': np.repeat(np.nan, 8), 'd': [np.nan, 3, 5, 7, 9, 11, 13, 15]}, dtype='Int64')), (1, DataFrame({'a': np.repeat(np.nan, 8), 'b': [0, 1, np.nan, 1, np.nan, np.nan, np.nan, 0], 'c': np.repeat(np.nan, 8), 'd': np.repeat(np.nan, 8)}, dtype='Int64'))])
def test_diff_integer_na(self, axis, expected):
    df = DataFrame({'a': np.repeat([0, 1, np.nan, 2], 2), 'b': np.tile([0, 1, np.nan, 2], 2), 'c': np.repeat(np.nan, 8), 'd': np.arange(1, 9) ** 2}, dtype='Int64')
    result = df.diff(axis=axis)
    tm.assert_frame_equal(result, expected)