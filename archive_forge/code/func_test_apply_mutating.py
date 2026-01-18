from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_mutating(using_array_manager, using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'a': range(100), 'b': range(100, 200)})
    df_orig = df.copy()

    def func(row):
        mgr = row._mgr
        row.loc['a'] += 1
        assert row._mgr is not mgr
        return row
    expected = df.copy()
    expected['a'] += 1
    with tm.assert_cow_warning(warn_copy_on_write):
        result = df.apply(func, axis=1)
    tm.assert_frame_equal(result, expected)
    if using_copy_on_write or using_array_manager:
        tm.assert_frame_equal(df, df_orig)
    else:
        tm.assert_frame_equal(df, result)