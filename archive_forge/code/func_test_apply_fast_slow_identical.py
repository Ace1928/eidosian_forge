from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_fast_slow_identical():
    df = DataFrame({'A': [0, 0, 1], 'b': range(3)})

    def slow(group):
        return group

    def fast(group):
        return group.copy()
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        fast_df = df.groupby('A', group_keys=False).apply(fast)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        slow_df = df.groupby('A', group_keys=False).apply(slow)
    tm.assert_frame_equal(fast_df, slow_df)