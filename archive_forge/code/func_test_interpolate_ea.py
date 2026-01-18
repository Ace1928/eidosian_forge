import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interpolate_ea(self, any_int_ea_dtype):
    df = DataFrame({'a': [1, None, None, None, 3]}, dtype=any_int_ea_dtype)
    orig = df.copy()
    result = df.interpolate(limit=2)
    expected = DataFrame({'a': [1, 1.5, 2.0, None, 3]}, dtype='Float64')
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(df, orig)