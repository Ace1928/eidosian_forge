import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('method', ['cummin', 'cummax'])
@pytest.mark.parametrize('dtype', ['float', 'Int64', 'Float64'])
@pytest.mark.parametrize('groups,expected_data', [([1, 1, 1], [1, None, None]), ([1, 2, 3], [1, None, 2]), ([1, 3, 3], [1, None, None])])
def test_cummin_max_skipna(method, dtype, groups, expected_data):
    df = DataFrame({'a': Series([1, None, 2], dtype=dtype)})
    orig = df.copy()
    gb = df.groupby(groups)['a']
    result = getattr(gb, method)(skipna=False)
    expected = Series(expected_data, dtype=dtype, name='a')
    tm.assert_frame_equal(df, orig)
    tm.assert_series_equal(result, expected)