from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('udf', [lambda x: x.copy(), lambda x: x.copy().rename(lambda y: y + 1)])
@pytest.mark.parametrize('group_keys', [True, False])
def test_apply_result_type(group_keys, udf):
    df = DataFrame({'A': ['a', 'b'], 'B': [1, 2]})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        df_result = df.groupby('A', group_keys=group_keys).apply(udf)
    series_result = df.B.groupby(df.A, group_keys=group_keys).apply(udf)
    if group_keys:
        assert df_result.index.nlevels == 2
        assert series_result.index.nlevels == 2
    else:
        assert df_result.index.nlevels == 1
        assert series_result.index.nlevels == 1