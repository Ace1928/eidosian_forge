import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_scalar(self, df):
    dtype = CategoricalDtype(list('cab'))
    result = df.loc['a']
    bidx = Series(list('aaa'), name='B').astype(dtype)
    assert bidx.dtype == dtype
    expected = DataFrame({'A': [0, 1, 5]}, index=Index(bidx))
    tm.assert_frame_equal(result, expected)
    df = df.copy()
    df.loc['a'] = 20
    bidx2 = Series(list('aabbca'), name='B').astype(dtype)
    assert bidx2.dtype == dtype
    expected = DataFrame({'A': [20, 20, 2, 3, 4, 20]}, index=Index(bidx2))
    tm.assert_frame_equal(df, expected)
    with pytest.raises(KeyError, match="^'d'$"):
        df.loc['d']
    df2 = df.copy()
    expected = df2.copy()
    expected.index = expected.index.astype(object)
    expected.loc['d'] = 10
    df2.loc['d'] = 10
    tm.assert_frame_equal(df2, expected)