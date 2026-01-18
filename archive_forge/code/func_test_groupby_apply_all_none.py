from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_apply_all_none():
    test_df = DataFrame({'groups': [0, 0, 1, 1], 'random_vars': [8, 7, 4, 5]})

    def test_func(x):
        pass
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = test_df.groupby('groups').apply(test_func)
    expected = DataFrame()
    tm.assert_frame_equal(result, expected)