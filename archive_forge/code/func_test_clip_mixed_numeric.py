import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_clip_mixed_numeric(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [1.0, np.nan, 3.0]})
    result = df.clip(1, 2)
    expected = DataFrame({'A': [1, 2, 2], 'B': [1.0, np.nan, 2.0]})
    tm.assert_frame_equal(result, expected)
    df = DataFrame([[1, 2, 3.4], [3, 4, 5.6]], columns=['foo', 'bar', 'baz'])
    expected = df.dtypes
    result = df.clip(upper=3).dtypes
    tm.assert_series_equal(result, expected)