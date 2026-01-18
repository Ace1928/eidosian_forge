from functools import partial
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_extension_array_dtype
def test_alignment_deprecation_enforced():
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'b': [1, 2, 3], 'c': [4, 5, 6]})
    s1 = pd.Series([1, 2], index=['a', 'b'])
    s2 = pd.Series([1, 2], index=['b', 'c'])
    expected = pd.DataFrame({'a': [2, 4, 6], 'b': [8, 10, 12]})
    with tm.assert_produces_warning(None):
        result = np.add(df1, df1)
    tm.assert_frame_equal(result, expected)
    result = np.add(df1, df2.values)
    tm.assert_frame_equal(result, expected)
    result = np.add(df1, df2)
    expected = pd.DataFrame({'a': [np.nan] * 3, 'b': [5, 7, 9], 'c': [np.nan] * 3})
    tm.assert_frame_equal(result, expected)
    result = np.add(df1.values, df2)
    expected = pd.DataFrame({'b': [2, 4, 6], 'c': [8, 10, 12]})
    tm.assert_frame_equal(result, expected)
    expected = pd.DataFrame({'a': [2, 3, 4], 'b': [6, 7, 8]})
    with tm.assert_produces_warning(None):
        result = np.add(df1, s1)
    tm.assert_frame_equal(result, expected)
    result = np.add(df1, s2.values)
    tm.assert_frame_equal(result, expected)
    expected = pd.DataFrame({'a': [np.nan] * 3, 'b': [5.0, 6.0, 7.0], 'c': [np.nan] * 3})
    result = np.add(df1, s2)
    tm.assert_frame_equal(result, expected)
    msg = "Cannot apply ufunc <ufunc 'add'> to mixed DataFrame and Series inputs."
    with pytest.raises(NotImplementedError, match=msg):
        np.add(s2, df1)