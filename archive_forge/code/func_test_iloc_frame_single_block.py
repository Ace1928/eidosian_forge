import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_iloc_frame_single_block(self, data):
    df = pd.DataFrame({'A': data})
    result = df.iloc[:, :]
    tm.assert_frame_equal(result, df)
    result = df.iloc[:, :1]
    tm.assert_frame_equal(result, df)
    result = df.iloc[:, :2]
    tm.assert_frame_equal(result, df)
    result = df.iloc[:, ::2]
    tm.assert_frame_equal(result, df)
    result = df.iloc[:, 1:2]
    tm.assert_frame_equal(result, df.iloc[:, :0])
    result = df.iloc[:, -1:]
    tm.assert_frame_equal(result, df)