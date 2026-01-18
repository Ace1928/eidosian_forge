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
def test_groupby_sum_below_mincount_nullable_integer():
    df = DataFrame({'a': [0, 1, 2], 'b': [0, 1, 2], 'c': [0, 1, 2]}, dtype='Int64')
    grouped = df.groupby('a')
    idx = Index([0, 1, 2], name='a', dtype='Int64')
    result = grouped['b'].sum(min_count=2)
    expected = Series([pd.NA] * 3, dtype='Int64', index=idx, name='b')
    tm.assert_series_equal(result, expected)
    result = grouped.sum(min_count=2)
    expected = DataFrame({'b': [pd.NA] * 3, 'c': [pd.NA] * 3}, dtype='Int64', index=idx)
    tm.assert_frame_equal(result, expected)