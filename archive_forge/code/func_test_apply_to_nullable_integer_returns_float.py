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
@pytest.mark.parametrize('values', [{'a': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'b': [1, pd.NA, 2, 1, pd.NA, 2, 1, pd.NA, 2]}, {'a': [1, 1, 2, 2, 3, 3], 'b': [1, 2, 1, 2, 1, 2]}])
@pytest.mark.parametrize('function', ['mean', 'median', 'var'])
def test_apply_to_nullable_integer_returns_float(values, function):
    output = 0.5 if function == 'var' else 1.5
    arr = np.array([output] * 3, dtype=float)
    idx = Index([1, 2, 3], name='a', dtype='Int64')
    expected = DataFrame({'b': arr}, index=idx).astype('Float64')
    groups = DataFrame(values, dtype='Int64').groupby('a')
    result = getattr(groups, function)()
    tm.assert_frame_equal(result, expected)
    result = groups.agg(function)
    tm.assert_frame_equal(result, expected)
    result = groups.agg([function])
    expected.columns = MultiIndex.from_tuples([('b', function)])
    tm.assert_frame_equal(result, expected)