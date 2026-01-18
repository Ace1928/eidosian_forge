from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('group_name', ['x', ['x']])
def test_groupby_axis_1(group_name):
    df = DataFrame(np.arange(12).reshape(3, 4), index=[0, 1, 0], columns=[10, 20, 10, 20])
    df.index.name = 'y'
    df.columns.name = 'x'
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(group_name, axis=1)
    results = gb.sum()
    expected = df.T.groupby(group_name).sum().T
    tm.assert_frame_equal(results, expected)
    iterables = [['bar', 'baz', 'foo'], ['one', 'two']]
    mi = MultiIndex.from_product(iterables=iterables, names=['x', 'x1'])
    df = DataFrame(np.arange(18).reshape(3, 6), index=[0, 1, 0], columns=mi)
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        gb = df.groupby(group_name, axis=1)
    results = gb.sum()
    expected = df.T.groupby(group_name).sum().T
    tm.assert_frame_equal(results, expected)