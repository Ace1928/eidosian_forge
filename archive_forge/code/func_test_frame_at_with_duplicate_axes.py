from datetime import (
import itertools
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_frame_at_with_duplicate_axes(self):
    arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
    df = DataFrame(arr, columns=['A', 'A'])
    result = df.at[0, 'A']
    expected = df.iloc[0].copy()
    tm.assert_series_equal(result, expected)
    result = df.T.at['A', 0]
    tm.assert_series_equal(result, expected)
    df.at[1, 'A'] = 2
    expected = Series([2.0, 2.0], index=['A', 'A'], name=1)
    tm.assert_series_equal(df.iloc[1], expected)