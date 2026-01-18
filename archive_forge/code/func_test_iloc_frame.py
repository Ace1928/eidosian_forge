import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_iloc_frame(self, data):
    df = pd.DataFrame({'A': data, 'B': np.arange(len(data), dtype='int64')})
    expected = pd.DataFrame({'A': data[:4]})
    result = df.iloc[:4, [0]]
    tm.assert_frame_equal(result, expected)
    result = df.iloc[[0, 1, 2, 3], [0]]
    tm.assert_frame_equal(result, expected)
    expected = pd.Series(data[:4], name='A')
    result = df.iloc[:4, 0]
    tm.assert_series_equal(result, expected)
    result = df.iloc[:4, 0]
    tm.assert_series_equal(result, expected)
    result = df.iloc[:, ::2]
    tm.assert_frame_equal(result, df[['A']])
    result = df[['B', 'A']].iloc[:, ::2]
    tm.assert_frame_equal(result, df[['B']])