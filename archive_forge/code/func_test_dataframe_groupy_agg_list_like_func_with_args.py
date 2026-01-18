import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_dataframe_groupy_agg_list_like_func_with_args():
    df = DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
    gb = df.groupby('y')

    def foo1(x, a=1, c=0):
        return x.sum() + a + c

    def foo2(x, b=2, c=0):
        return x.sum() + b + c
    msg = "foo1\\(\\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        gb.agg([foo1, foo2], 3, b=3, c=4)
    result = gb.agg([foo1, foo2], 3, c=4)
    expected = DataFrame([[8, 8], [9, 9], [10, 10]], index=Index(['a', 'b', 'c'], name='y'), columns=MultiIndex.from_tuples([('x', 'foo1'), ('x', 'foo2')]))
    tm.assert_frame_equal(result, expected)