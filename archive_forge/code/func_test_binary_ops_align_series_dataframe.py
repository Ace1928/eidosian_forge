from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_binary_ops_align_series_dataframe(self):
    midx = MultiIndex.from_product([['A', 'B'], ['a', 'b']])
    df = DataFrame(np.ones((2, 4), dtype='int64'), columns=midx)
    s = Series({'a': 1, 'b': 2})
    df2 = df.copy()
    df2.columns.names = ['lvl0', 'lvl1']
    s2 = s.copy()
    s2.index.name = 'lvl1'
    res1 = df.mul(s, axis=1, level=1)
    res2 = df.mul(s2, axis=1, level=1)
    res3 = df2.mul(s, axis=1, level=1)
    res4 = df2.mul(s2, axis=1, level=1)
    res5 = df2.mul(s, axis=1, level='lvl1')
    res6 = df2.mul(s2, axis=1, level='lvl1')
    exp = DataFrame(np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype='int64'), columns=midx)
    for res in [res1, res2]:
        tm.assert_frame_equal(res, exp)
    exp.columns.names = ['lvl0', 'lvl1']
    for res in [res3, res4, res5, res6]:
        tm.assert_frame_equal(res, exp)