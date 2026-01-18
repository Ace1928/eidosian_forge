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
def test_dataframe_dummies_with_categorical(self, df, sparse, dtype):
    df['cat'] = Categorical(['x', 'y', 'y'])
    result = get_dummies(df, sparse=sparse, dtype=dtype).sort_index(axis=1)
    if sparse:
        arr = SparseArray
        if dtype.kind == 'b':
            typ = SparseDtype(dtype, False)
        else:
            typ = SparseDtype(dtype, 0)
    else:
        arr = np.array
        typ = dtype
    expected = DataFrame({'C': [1, 2, 3], 'A_a': arr([1, 0, 1], dtype=typ), 'A_b': arr([0, 1, 0], dtype=typ), 'B_b': arr([1, 1, 0], dtype=typ), 'B_c': arr([0, 0, 1], dtype=typ), 'cat_x': arr([1, 0, 0], dtype=typ), 'cat_y': arr([0, 1, 1], dtype=typ)}).sort_index(axis=1)
    tm.assert_frame_equal(result, expected)