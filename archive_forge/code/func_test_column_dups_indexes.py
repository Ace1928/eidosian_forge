import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_column_dups_indexes(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), index=['a', 'b', 'c', 'd', 'e'], columns=['A', 'B', 'A'])
    for index in [df.index, pd.Index(list('edcba'))]:
        this_df = df.copy()
        expected_ser = Series(index.values, index=this_df.index)
        expected_df = DataFrame({'A': expected_ser, 'B': this_df['B']}, columns=['A', 'B', 'A'])
        this_df['A'] = index
        tm.assert_frame_equal(this_df, expected_df)