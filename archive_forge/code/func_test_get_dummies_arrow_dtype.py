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
@td.skip_if_no('pyarrow')
def test_get_dummies_arrow_dtype(self):
    df = DataFrame({'name': Series(['a'], dtype=ArrowDtype(pa.string())), 'x': 1})
    result = get_dummies(df)
    expected = DataFrame({'x': 1, 'name_a': Series([True], dtype='bool[pyarrow]')})
    tm.assert_frame_equal(result, expected)
    df = DataFrame({'name': Series(['a'], dtype=CategoricalDtype(Index(['a'], dtype=ArrowDtype(pa.string())))), 'x': 1})
    result = get_dummies(df)
    tm.assert_frame_equal(result, expected)