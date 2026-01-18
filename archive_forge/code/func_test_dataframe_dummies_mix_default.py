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
def test_dataframe_dummies_mix_default(self, df, sparse, dtype):
    result = get_dummies(df, sparse=sparse, dtype=dtype)
    if sparse:
        arr = SparseArray
        if dtype.kind == 'b':
            typ = SparseDtype(dtype, False)
        else:
            typ = SparseDtype(dtype, 0)
    else:
        arr = np.array
        typ = dtype
    expected = DataFrame({'C': [1, 2, 3], 'A_a': arr([1, 0, 1], dtype=typ), 'A_b': arr([0, 1, 0], dtype=typ), 'B_b': arr([1, 1, 0], dtype=typ), 'B_c': arr([0, 0, 1], dtype=typ)})
    expected = expected[['C', 'A_a', 'A_b', 'B_b', 'B_c']]
    tm.assert_frame_equal(result, expected)