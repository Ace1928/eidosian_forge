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
def test_dataframe_dummies_all_obj(self, df, sparse):
    df = df[['A', 'B']]
    result = get_dummies(df, sparse=sparse)
    expected = DataFrame({'A_a': [1, 0, 1], 'A_b': [0, 1, 0], 'B_b': [1, 1, 0], 'B_c': [0, 0, 1]}, dtype=bool)
    if sparse:
        expected = DataFrame({'A_a': SparseArray([1, 0, 1], dtype='bool'), 'A_b': SparseArray([0, 1, 0], dtype='bool'), 'B_b': SparseArray([1, 1, 0], dtype='bool'), 'B_c': SparseArray([0, 0, 1], dtype='bool')})
    tm.assert_frame_equal(result, expected)