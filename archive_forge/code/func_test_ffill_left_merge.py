import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ffill_left_merge(self):
    df1 = DataFrame({'key': ['a', 'c', 'e', 'a', 'c', 'e'], 'lvalue': [1, 2, 3, 1, 2, 3], 'group': ['a', 'a', 'a', 'b', 'b', 'b']})
    df2 = DataFrame({'key': ['b', 'c', 'd'], 'rvalue': [1, 2, 3]})
    result = merge_ordered(df1, df2, fill_method='ffill', left_by='group', how='left')
    expected = DataFrame({'key': ['a', 'c', 'e', 'a', 'c', 'e'], 'lvalue': [1, 2, 3, 1, 2, 3], 'group': ['a', 'a', 'a', 'b', 'b', 'b'], 'rvalue': [np.nan, 2.0, 2.0, np.nan, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)