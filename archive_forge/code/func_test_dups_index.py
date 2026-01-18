from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dups_index(self):
    df = DataFrame(np.random.default_rng(2).integers(0, 10, size=40).reshape(10, 4), columns=['A', 'A', 'C', 'C'])
    result = concat([df, df], axis=1)
    tm.assert_frame_equal(result.iloc[:, :4], df)
    tm.assert_frame_equal(result.iloc[:, 4:], df)
    result = concat([df, df], axis=0)
    tm.assert_frame_equal(result.iloc[:10], df)
    tm.assert_frame_equal(result.iloc[10:], df)
    df = concat([DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=['A', 'A', 'B', 'B']), DataFrame(np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2), columns=['A', 'C'])], axis=1)
    result = concat([df, df], axis=1)
    tm.assert_frame_equal(result.iloc[:, :6], df)
    tm.assert_frame_equal(result.iloc[:, 6:], df)
    result = concat([df, df], axis=0)
    tm.assert_frame_equal(result.iloc[:10], df)
    tm.assert_frame_equal(result.iloc[10:], df)
    result = df.iloc[0:8, :]._append(df.iloc[8:])
    tm.assert_frame_equal(result, df)
    result = df.iloc[0:8, :]._append(df.iloc[8:9])._append(df.iloc[9:10])
    tm.assert_frame_equal(result, df)
    expected = concat([df, df], axis=0)
    result = df._append(df)
    tm.assert_frame_equal(result, expected)