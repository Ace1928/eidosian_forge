import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dup_columns_comparisons(self):
    df1 = DataFrame([[1, 2], [2, np.nan], [3, 4], [4, 4]], columns=['A', 'B'])
    df2 = DataFrame([[0, 1], [2, 4], [2, np.nan], [4, 5]], columns=['A', 'A'])
    msg = 'Can only compare identically-labeled \\(both index and columns\\) DataFrame objects'
    with pytest.raises(ValueError, match=msg):
        df1 == df2
    df1r = df1.reindex_like(df2)
    result = df1r == df2
    expected = DataFrame([[False, True], [True, False], [False, False], [True, False]], columns=['A', 'A'])
    tm.assert_frame_equal(result, expected)