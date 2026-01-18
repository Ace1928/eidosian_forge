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
@pytest.mark.parametrize('op', [lambda x: x.sum(), lambda x: x.mean()])
def test_groupby_multiple_columns(df, op):
    data = df
    grouped = data.groupby(['A', 'B'])
    result1 = op(grouped)
    keys = []
    values = []
    for n1, gp1 in data.groupby('A'):
        for n2, gp2 in gp1.groupby('B'):
            keys.append((n1, n2))
            values.append(op(gp2.loc[:, ['C', 'D']]))
    mi = MultiIndex.from_tuples(keys, names=['A', 'B'])
    expected = pd.concat(values, axis=1).T
    expected.index = mi
    for col in ['C', 'D']:
        result_col = op(grouped[col])
        pivoted = result1[col]
        exp = expected[col]
        tm.assert_series_equal(result_col, exp)
        tm.assert_series_equal(pivoted, exp)
    result = data['C'].groupby([data['A'], data['B']]).mean()
    expected = data.groupby(['A', 'B']).mean()['C']
    tm.assert_series_equal(result, expected)