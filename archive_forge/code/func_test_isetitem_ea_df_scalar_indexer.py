import pytest
from pandas import (
import pandas._testing as tm
def test_isetitem_ea_df_scalar_indexer(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6]])
    rhs = DataFrame([[11], [13]], dtype='Int64')
    df.isetitem(2, rhs)
    expected = DataFrame({0: [1, 4], 1: [2, 5], 2: Series([11, 13], dtype='Int64')})
    tm.assert_frame_equal(df, expected)