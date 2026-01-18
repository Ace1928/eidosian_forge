import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('how', ['idxmax', 'idxmin'])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_idxmin_idxmax_transform_args(how, skipna, numeric_only):
    df = DataFrame({'a': [1, 1, 1, 2], 'b': [3.0, 4.0, np.nan, 6.0], 'c': list('abcd')})
    gb = df.groupby('a')
    msg = f"'axis' keyword in DataFrameGroupBy.{how} is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = gb.transform(how, 0, skipna, numeric_only)
    warn = None if skipna else FutureWarning
    msg = f'The behavior of DataFrameGroupBy.{how} with .* any-NA and skipna=False'
    with tm.assert_produces_warning(warn, match=msg):
        expected = gb.transform(how, skipna=skipna, numeric_only=numeric_only)
    tm.assert_frame_equal(result, expected)