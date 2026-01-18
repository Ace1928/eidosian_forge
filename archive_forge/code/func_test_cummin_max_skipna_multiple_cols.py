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
def test_cummin_max_skipna_multiple_cols(method):
    df = DataFrame({'a': [np.nan, 2.0, 2.0], 'b': [2.0, 2.0, 2.0]})
    gb = df.groupby([1, 1, 1])[['a', 'b']]
    result = getattr(gb, method)(skipna=False)
    expected = DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [2.0, 2.0, 2.0]})
    tm.assert_frame_equal(result, expected)