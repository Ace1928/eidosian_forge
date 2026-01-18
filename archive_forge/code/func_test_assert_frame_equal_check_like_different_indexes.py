import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_assert_frame_equal_check_like_different_indexes():
    df1 = DataFrame(index=pd.Index([], dtype='object'))
    df2 = DataFrame(index=pd.RangeIndex(start=0, stop=0, step=1))
    with pytest.raises(AssertionError, match='DataFrame.index are different'):
        tm.assert_frame_equal(df1, df2, check_like=True)