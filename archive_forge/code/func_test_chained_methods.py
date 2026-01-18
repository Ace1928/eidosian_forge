import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('method, idx', [(lambda df: df.copy(deep=False).copy(deep=False), 0), (lambda df: df.reset_index().reset_index(), 2), (lambda df: df.rename(columns=str.upper).rename(columns=str.lower), 0), (lambda df: df.copy(deep=False).select_dtypes(include='number'), 0)], ids=['shallow-copy', 'reset_index', 'rename', 'select_dtypes'])
def test_chained_methods(request, method, idx, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [0.1, 0.2, 0.3]})
    df_orig = df.copy()
    df2_is_view = not using_copy_on_write and request.node.callspec.id == 'shallow-copy'
    df2 = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and df2_is_view):
        df2.iloc[0, idx] = 0
    if not df2_is_view:
        tm.assert_frame_equal(df, df_orig)
    df2 = method(df)
    with tm.assert_cow_warning(warn_copy_on_write and df2_is_view):
        df.iloc[0, 0] = 0
    if not df2_is_view:
        tm.assert_frame_equal(df2.iloc[:, idx:], df_orig)