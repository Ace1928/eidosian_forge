import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_rowwise(self):
    df = DataFrame({0: [1, 2, np.nan, 4], 1: [2, 3, 4, np.nan], 2: [np.nan, 4, 5, 6], 3: [4, np.nan, 6, 7], 4: [1, 2, 3, 4]})
    result = df.interpolate(axis=1)
    expected = df.copy()
    expected.loc[3, 1] = 5
    expected.loc[0, 2] = 3
    expected.loc[1, 3] = 3
    expected[4] = expected[4].astype(np.float64)
    tm.assert_frame_equal(result, expected)
    result = df.interpolate(axis=1, method='values')
    tm.assert_frame_equal(result, expected)
    result = df.interpolate(axis=0)
    expected = df.interpolate()
    tm.assert_frame_equal(result, expected)