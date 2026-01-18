import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_ignore_extension_dtype_mismatch_cross_class():
    left = DataFrame({'a': [1, 2, 3]}, dtype='Int64')
    right = DataFrame({'a': [1, 2, 3]}, dtype='int64')
    tm.assert_frame_equal(left, right, check_dtype=False)