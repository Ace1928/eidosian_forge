import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_ea_column_definition_in_exception():
    df1 = DataFrame({'a': pd.Series([pd.NA, 1], dtype='Int64')})
    df2 = DataFrame({'a': pd.Series([pd.NA, 2], dtype='Int64')})
    msg = 'DataFrame.iloc\\[:, 0\\] \\(column name="a"\\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, check_exact=True)