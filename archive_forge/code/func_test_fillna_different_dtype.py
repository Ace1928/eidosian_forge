import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_different_dtype(self, using_infer_string):
    df = DataFrame([['a', 'a', np.nan, 'a'], ['b', 'b', np.nan, 'b'], ['c', 'c', np.nan, 'c']])
    if using_infer_string:
        with tm.assert_produces_warning(FutureWarning, match='Downcasting'):
            result = df.fillna({2: 'foo'})
    else:
        result = df.fillna({2: 'foo'})
    expected = DataFrame([['a', 'a', 'foo', 'a'], ['b', 'b', 'foo', 'b'], ['c', 'c', 'foo', 'c']])
    tm.assert_frame_equal(result, expected)
    if using_infer_string:
        with tm.assert_produces_warning(FutureWarning, match='Downcasting'):
            return_value = df.fillna({2: 'foo'}, inplace=True)
    else:
        return_value = df.fillna({2: 'foo'}, inplace=True)
    tm.assert_frame_equal(df, expected)
    assert return_value is None