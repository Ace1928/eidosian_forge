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
@pytest.mark.parametrize('data, groups', [([0, 1, 2, 3], [0, 0, 1, 1]), ([0], [0])])
@pytest.mark.parametrize('dtype', [None, *tm.ALL_INT_NUMPY_DTYPES])
@pytest.mark.parametrize('method', ['nlargest', 'nsmallest'])
def test_nlargest_and_smallest_noop(data, groups, dtype, method):
    if dtype is not None:
        data = np.array(data, dtype=dtype)
    if method == 'nlargest':
        data = list(reversed(data))
    ser = Series(data, name='a')
    result = getattr(ser.groupby(groups), method)(n=2)
    expidx = np.array(groups, dtype=np.int_) if isinstance(groups, list) else groups
    expected = Series(data, index=MultiIndex.from_arrays([expidx, ser.index]), name='a')
    tm.assert_series_equal(result, expected)