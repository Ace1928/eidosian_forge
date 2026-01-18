import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_loc_frame(self, data):
    df = pd.DataFrame({'A': data, 'B': np.arange(len(data), dtype='int64')})
    expected = pd.DataFrame({'A': data[:4]})
    result = df.loc[:3, ['A']]
    tm.assert_frame_equal(result, expected)
    result = df.loc[[0, 1, 2, 3], ['A']]
    tm.assert_frame_equal(result, expected)
    expected = pd.Series(data[:4], name='A')
    result = df.loc[:3, 'A']
    tm.assert_series_equal(result, expected)
    result = df.loc[:3, 'A']
    tm.assert_series_equal(result, expected)