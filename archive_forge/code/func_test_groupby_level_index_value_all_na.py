from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_level_index_value_all_na(self):
    df = DataFrame([['x', np.nan, 10], [None, np.nan, 20]], columns=['A', 'B', 'C']).set_index(['A', 'B'])
    result = df.groupby(level=['A', 'B']).sum()
    expected = DataFrame(data=[], index=MultiIndex(levels=[Index(['x'], dtype='object'), Index([], dtype='float64')], codes=[[], []], names=['A', 'B']), columns=['C'], dtype='int64')
    tm.assert_frame_equal(result, expected)