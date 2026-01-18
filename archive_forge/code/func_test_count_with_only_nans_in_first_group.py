from itertools import product
from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_count_with_only_nans_in_first_group(self):
    df = DataFrame({'A': [np.nan, np.nan], 'B': ['a', 'b'], 'C': [1, 2]})
    result = df.groupby(['A', 'B']).C.count()
    mi = MultiIndex(levels=[[], ['a', 'b']], codes=[[], []], names=['A', 'B'])
    expected = Series([], index=mi, dtype=np.int64, name='C')
    tm.assert_series_equal(result, expected, check_index_type=False)