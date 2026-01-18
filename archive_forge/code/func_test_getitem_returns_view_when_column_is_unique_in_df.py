import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_returns_view_when_column_is_unique_in_df(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
    df_orig = df.copy()
    view = df['b']
    with tm.assert_cow_warning(warn_copy_on_write):
        view.loc[:] = 100
    if using_copy_on_write:
        expected = df_orig
    else:
        expected = DataFrame([[1, 2, 100], [4, 5, 100]], columns=['a', 'a', 'b'])
    tm.assert_frame_equal(df, expected)