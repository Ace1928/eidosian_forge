import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_clip_against_unordered_columns(self):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((1000, 4)), columns=['A', 'B', 'C', 'D'])
    df2 = DataFrame(np.random.default_rng(2).standard_normal((1000, 4)), columns=['D', 'A', 'B', 'C'])
    df3 = DataFrame(df2.values - 1, columns=['B', 'D', 'C', 'A'])
    result_upper = df1.clip(lower=0, upper=df2)
    expected_upper = df1.clip(lower=0, upper=df2[df1.columns])
    result_lower = df1.clip(lower=df3, upper=3)
    expected_lower = df1.clip(lower=df3[df1.columns], upper=3)
    result_lower_upper = df1.clip(lower=df3, upper=df2)
    expected_lower_upper = df1.clip(lower=df3[df1.columns], upper=df2[df1.columns])
    tm.assert_frame_equal(result_upper, expected_upper)
    tm.assert_frame_equal(result_lower, expected_lower)
    tm.assert_frame_equal(result_lower_upper, expected_lower_upper)