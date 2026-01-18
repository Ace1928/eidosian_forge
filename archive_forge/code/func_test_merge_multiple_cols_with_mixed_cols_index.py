import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_merge_multiple_cols_with_mixed_cols_index(self):
    s = Series(range(6), MultiIndex.from_product([['A', 'B'], [1, 2, 3]], names=['lev1', 'lev2']), name='Amount')
    df = DataFrame({'lev1': list('AAABBB'), 'lev2': [1, 2, 3, 1, 2, 3], 'col': 0})
    result = merge(df, s.reset_index(), on=['lev1', 'lev2'])
    expected = DataFrame({'lev1': list('AAABBB'), 'lev2': [1, 2, 3, 1, 2, 3], 'col': [0] * 6, 'Amount': range(6)})
    tm.assert_frame_equal(result, expected)