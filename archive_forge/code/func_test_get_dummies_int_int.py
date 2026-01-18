import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_get_dummies_int_int(self):
    data = Series([1, 2, 1])
    result = get_dummies(data)
    expected = DataFrame([[1, 0], [0, 1], [1, 0]], columns=[1, 2], dtype=bool)
    tm.assert_frame_equal(result, expected)
    data = Series(Categorical(['a', 'b', 'a']))
    result = get_dummies(data)
    expected = DataFrame([[1, 0], [0, 1], [1, 0]], columns=Categorical(['a', 'b']), dtype=bool)
    tm.assert_frame_equal(result, expected)