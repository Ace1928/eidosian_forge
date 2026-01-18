import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_multiple_frames_dtypes(self):
    df1 = DataFrame(data=np.ones((10, 2)), columns=['foo', 'bar'], dtype=np.float64)
    df2 = DataFrame(data=np.ones((10, 2)), dtype=np.float32)
    results = concat((df1, df2), axis=1).dtypes
    expected = Series([np.dtype('float64')] * 2 + [np.dtype('float32')] * 2, index=['foo', 'bar', 0, 1])
    tm.assert_series_equal(results, expected)