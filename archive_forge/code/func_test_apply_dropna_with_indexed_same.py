from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('dropna', [True, False])
def test_apply_dropna_with_indexed_same(dropna):
    df = DataFrame({'col': [1, 2, 3, 4, 5], 'group': ['a', np.nan, np.nan, 'b', 'b']}, index=list('xxyxz'))
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('group', dropna=dropna, group_keys=False).apply(lambda x: x)
    expected = df.dropna() if dropna else df.iloc[[0, 3, 1, 2, 4]]
    tm.assert_frame_equal(result, expected)