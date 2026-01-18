import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_unsorted_order(self):
    df = DataFrame({'b': [3, 1, 2], 'a': [5, 4, 6]}, index=['C', 'A', 'B'])
    result = crosstab(df.index, [df.b, df.a])
    e_idx = Index(['A', 'B', 'C'], name='row_0')
    e_columns = MultiIndex.from_tuples([(1, 4), (2, 6), (3, 5)], names=['b', 'a'])
    expected = DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1]], index=e_idx, columns=e_columns)
    tm.assert_frame_equal(result, expected)