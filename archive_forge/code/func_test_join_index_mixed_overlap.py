import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_index_mixed_overlap(self):
    df1 = DataFrame({'A': 1.0, 'B': 2, 'C': 'foo', 'D': True}, index=np.arange(10), columns=['A', 'B', 'C', 'D'])
    assert df1['B'].dtype == np.int64
    assert df1['D'].dtype == np.bool_
    df2 = DataFrame({'A': 1.0, 'B': 2, 'C': 'foo', 'D': True}, index=np.arange(0, 10, 2), columns=['A', 'B', 'C', 'D'])
    joined = df1.join(df2, lsuffix='_one', rsuffix='_two')
    expected_columns = ['A_one', 'B_one', 'C_one', 'D_one', 'A_two', 'B_two', 'C_two', 'D_two']
    df1.columns = expected_columns[:4]
    df2.columns = expected_columns[4:]
    expected = _join_by_hand(df1, df2)
    tm.assert_frame_equal(joined, expected)