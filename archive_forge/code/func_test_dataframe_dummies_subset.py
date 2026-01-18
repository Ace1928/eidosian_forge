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
def test_dataframe_dummies_subset(self, df, sparse):
    result = get_dummies(df, prefix=['from_A'], columns=['A'], sparse=sparse)
    expected = DataFrame({'B': ['b', 'b', 'c'], 'C': [1, 2, 3], 'from_A_a': [1, 0, 1], 'from_A_b': [0, 1, 0]})
    cols = expected.columns
    expected[cols[1:]] = expected[cols[1:]].astype(bool)
    expected[['C']] = df[['C']]
    if sparse:
        cols = ['from_A_a', 'from_A_b']
        expected[cols] = expected[cols].astype(SparseDtype('bool', False))
    tm.assert_frame_equal(result, expected)