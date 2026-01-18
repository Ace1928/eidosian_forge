import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_set_mismatch():
    df1 = DataFrame({'set_column': [{1, 2, 3}, {4, 5, 6}]})
    df2 = DataFrame({'set_column': [{1, 2, 3}, {4, 5, 7}]})
    msg = 'DataFrame.iloc\\[:, 0\\] \\(column name="set_column"\\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)