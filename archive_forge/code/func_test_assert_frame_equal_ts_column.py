import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_ts_column():
    df1 = DataFrame({'a': [pd.Timestamp('2019-12-31'), pd.Timestamp('2020-12-31')]})
    df2 = DataFrame({'a': [pd.Timestamp('2020-12-31'), pd.Timestamp('2020-12-31')]})
    msg = 'DataFrame.iloc\\[:, 0\\] \\(column name="a"\\) values are different'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)