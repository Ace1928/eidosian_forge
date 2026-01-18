from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('index', [pd.CategoricalIndex(list('abc')), pd.interval_range(0, 3), pd.period_range('2020', periods=3, freq='D'), MultiIndex.from_tuples([('a', 0), ('a', 1), ('b', 0)])])
def test_apply_index_has_complex_internals(index):
    df = DataFrame({'group': [1, 1, 2], 'value': [0, 1, 0]}, index=index)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('group', group_keys=False).apply(lambda x: x)
    tm.assert_frame_equal(result, df)