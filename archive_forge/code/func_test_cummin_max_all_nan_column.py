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
@pytest.mark.parametrize('dtype', ['UInt64', 'Int64', 'Float64', 'float', 'boolean'])
def test_cummin_max_all_nan_column(method, dtype):
    base_df = DataFrame({'A': [1, 1, 1, 1, 2, 2, 2, 2], 'B': [np.nan] * 8})
    base_df['B'] = base_df['B'].astype(dtype)
    grouped = base_df.groupby('A')
    expected = DataFrame({'B': [np.nan] * 8}, dtype=dtype)
    result = getattr(grouped, method)()
    tm.assert_frame_equal(expected, result)
    result = getattr(grouped['B'], method)().to_frame()
    tm.assert_frame_equal(expected, result)