import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_cumsum_preserve_dtypes(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [1, 2, 3.0], 'C': [True, False, False]})
    result = df.cumsum()
    expected = DataFrame({'A': Series([1, 3, 6], dtype=np.int64), 'B': Series([1, 3, 6], dtype=np.float64), 'C': df['C'].cumsum()})
    tm.assert_frame_equal(result, expected)