from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_apply_return_empty_chunk():
    df = DataFrame({'value': [0, 1], 'group': ['filled', 'empty']})
    groups = df.groupby('group')
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = groups.apply(lambda group: group[group.value != 1]['value'])
    expected = Series([0], name='value', index=MultiIndex.from_product([['empty', 'filled'], [0]], names=['group', None]).drop('empty'))
    tm.assert_series_equal(result, expected)