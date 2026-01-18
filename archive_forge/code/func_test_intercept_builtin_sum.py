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
def test_intercept_builtin_sum():
    s = Series([1.0, 2.0, np.nan, 3.0])
    grouped = s.groupby([0, 1, 2, 2])
    result = grouped.agg(builtins.sum)
    result2 = grouped.apply(builtins.sum)
    expected = grouped.sum()
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result2, expected)