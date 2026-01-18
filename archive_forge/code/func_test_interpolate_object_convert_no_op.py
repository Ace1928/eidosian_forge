import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_interpolate_object_convert_no_op(using_copy_on_write):
    df = DataFrame({'a': ['a', 'b', 'c'], 'b': 1})
    arr_a = get_array(df, 'a')
    msg = 'DataFrame.interpolate with method=pad is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.interpolate(method='pad', inplace=True)
    if using_copy_on_write:
        assert df._mgr._has_no_reference(0)
        assert np.shares_memory(arr_a, get_array(df, 'a'))