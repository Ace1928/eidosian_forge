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
def test_corrwith_with_1_axis():
    df = DataFrame({'a': [1, 1, 2], 'b': [3, 7, 4]})
    result = df.groupby('a').corrwith(df, axis=1)
    index = Index(data=[(1, 0), (1, 1), (1, 2), (2, 2), (2, 0), (2, 1)], name=('a', None))
    expected = Series([np.nan] * 6, index=index)
    tm.assert_series_equal(result, expected)