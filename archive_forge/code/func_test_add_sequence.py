import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_add_sequence(dtype):
    a = pd.array(['a', 'b', None, None], dtype=dtype)
    other = ['x', None, 'y', None]
    result = a + other
    expected = pd.array(['ax', None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)
    result = other + a
    expected = pd.array(['xa', None, None, None], dtype=dtype)
    tm.assert_extension_array_equal(result, expected)