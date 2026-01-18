from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_frame_not_as_index_column_name(df):
    grouped = df.groupby(['A', 'B'], as_index=False)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = grouped.apply(len)
    expected = grouped.count().rename(columns={'C': np.nan}).drop(columns='D')
    tm.assert_index_equal(result.index, expected.index)
    tm.assert_numpy_array_equal(result.values, expected.values)