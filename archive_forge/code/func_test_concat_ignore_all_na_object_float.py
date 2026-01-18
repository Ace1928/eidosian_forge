from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
@td.skip_array_manager_invalid_test
@pytest.mark.parametrize('df_dtype', ['float64', 'int64', 'datetime64[ns]'])
@pytest.mark.parametrize('empty_dtype', [None, 'float64', 'object'])
def test_concat_ignore_all_na_object_float(empty_dtype, df_dtype):
    df = DataFrame({'foo': [1, 2], 'bar': [1, 2]}, dtype=df_dtype)
    empty = DataFrame({'foo': [np.nan], 'bar': [np.nan]}, dtype=empty_dtype)
    if df_dtype == 'int64':
        if empty_dtype == 'object':
            df_dtype = 'object'
        else:
            df_dtype = 'float64'
    msg = 'The behavior of DataFrame concatenation with empty or all-NA entries'
    warn = None
    if empty_dtype != df_dtype and empty_dtype is not None:
        warn = FutureWarning
    elif df_dtype == 'datetime64[ns]':
        warn = FutureWarning
    with tm.assert_produces_warning(warn, match=msg):
        result = concat([empty, df], ignore_index=True)
    expected = DataFrame({'foo': [np.nan, 1, 2], 'bar': [np.nan, 1, 2]}, dtype=df_dtype)
    tm.assert_frame_equal(result, expected)