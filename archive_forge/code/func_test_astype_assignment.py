import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_astype_assignment(self, using_infer_string):
    df_orig = DataFrame([['1', '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
    df = df_orig.copy()
    df.iloc[:, 0:2] = df.iloc[:, 0:2].astype(np.int64)
    expected = DataFrame([[1, 2, '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
    if not using_infer_string:
        expected['A'] = expected['A'].astype(object)
        expected['B'] = expected['B'].astype(object)
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[:, 'A'] = df.loc[:, 'A'].astype(np.int64)
    expected = DataFrame([[1, '2', '3', '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
    if not using_infer_string:
        expected['A'] = expected['A'].astype(object)
    tm.assert_frame_equal(df, expected)
    df = df_orig.copy()
    df.loc[:, ['B', 'C']] = df.loc[:, ['B', 'C']].astype(np.int64)
    expected = DataFrame([['1', 2, 3, '.4', 5, 6.0, 'foo']], columns=list('ABCDEFG'))
    if not using_infer_string:
        expected['B'] = expected['B'].astype(object)
        expected['C'] = expected['C'].astype(object)
    tm.assert_frame_equal(df, expected)