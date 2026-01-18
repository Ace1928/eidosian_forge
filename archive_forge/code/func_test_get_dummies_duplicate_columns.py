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
def test_get_dummies_duplicate_columns(self, df):
    df.columns = ['A', 'A', 'A']
    result = get_dummies(df).sort_index(axis=1)
    expected = DataFrame([[1, True, False, True, False], [2, False, True, True, False], [3, True, False, False, True]], columns=['A', 'A_a', 'A_b', 'A_b', 'A_c']).sort_index(axis=1)
    expected = expected.astype({'A': np.int64})
    tm.assert_frame_equal(result, expected)