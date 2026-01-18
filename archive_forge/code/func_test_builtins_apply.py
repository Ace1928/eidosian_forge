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
@pytest.mark.parametrize('f', [max, min, sum])
@pytest.mark.parametrize('keys', ['jim', ['jim', 'joe']])
def test_builtins_apply(keys, f):
    df = DataFrame(np.random.randint(1, 50, (1000, 2)), columns=['jim', 'joe'])
    df['jolie'] = np.random.randn(1000)
    gb = df.groupby(keys)
    fname = f.__name__
    result = gb.apply(f)
    ngroups = len(df.drop_duplicates(subset=keys))
    assert_msg = f'invalid frame shape: {result.shape} (expected ({ngroups}, 3))'
    assert result.shape == (ngroups, 3), assert_msg
    npfunc = lambda x: getattr(np, fname)(x, axis=0)
    expected = gb.apply(npfunc)
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(None):
        expected2 = gb.apply(lambda x: npfunc(x))
    tm.assert_frame_equal(result, expected2)
    if f != sum:
        expected = gb.agg(fname).reset_index()
        expected.set_index(keys, inplace=True, drop=False)
        tm.assert_frame_equal(result, expected, check_dtype=False)
    tm.assert_series_equal(getattr(result, fname)(axis=0), getattr(df, fname)(axis=0))