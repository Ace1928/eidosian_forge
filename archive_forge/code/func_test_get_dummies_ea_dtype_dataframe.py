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
def test_get_dummies_ea_dtype_dataframe(self, any_numeric_ea_and_arrow_dtype):
    df = DataFrame({'x': list('abca')})
    result = get_dummies(df, dtype=any_numeric_ea_and_arrow_dtype)
    expected = DataFrame({'x_a': [1, 0, 0, 1], 'x_b': [0, 1, 0, 0], 'x_c': [0, 0, 1, 0]}, dtype=any_numeric_ea_and_arrow_dtype)
    tm.assert_frame_equal(result, expected)