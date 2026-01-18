import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_compare_tuple_strs():
    mi = MultiIndex.from_tuples([('a', 'b'), ('b', 'c'), ('c', 'a')])
    result = mi == ('c', 'a')
    expected = np.array([False, False, True])
    tm.assert_numpy_array_equal(result, expected)
    result = mi == ('c',)
    expected = np.array([False, False, False])
    tm.assert_numpy_array_equal(result, expected)