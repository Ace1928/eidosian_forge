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
def test_dataframe_dummies_with_na(self, df, sparse, dtype):
    df.loc[3, :] = [np.nan, np.nan, np.nan]
    result = get_dummies(df, dummy_na=True, sparse=sparse, dtype=dtype).sort_index(axis=1)
    if sparse:
        arr = SparseArray
        if dtype.kind == 'b':
            typ = SparseDtype(dtype, False)
        else:
            typ = SparseDtype(dtype, 0)
    else:
        arr = np.array
        typ = dtype
    expected = DataFrame({'C': [1, 2, 3, np.nan], 'A_a': arr([1, 0, 1, 0], dtype=typ), 'A_b': arr([0, 1, 0, 0], dtype=typ), 'A_nan': arr([0, 0, 0, 1], dtype=typ), 'B_b': arr([1, 1, 0, 0], dtype=typ), 'B_c': arr([0, 0, 1, 0], dtype=typ), 'B_nan': arr([0, 0, 0, 1], dtype=typ)}).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)
    result = get_dummies(df, dummy_na=False, sparse=sparse, dtype=dtype)
    expected = expected[['C', 'A_a', 'A_b', 'B_b', 'B_c']]
    tm.assert_frame_equal(result, expected)