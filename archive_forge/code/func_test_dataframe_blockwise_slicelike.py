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
def test_dataframe_blockwise_slicelike():
    arr = np.random.default_rng(2).integers(0, 1000, (100, 10))
    df1 = DataFrame(arr)
    df2 = df1.copy().astype({1: 'float', 3: 'float', 7: 'float'})
    df2.iloc[0, [1, 3, 7]] = np.nan
    df3 = df1.copy().astype({5: 'float'})
    df3.iloc[0, [5]] = np.nan
    df4 = df1.copy().astype({2: 'float', 3: 'float', 4: 'float'})
    df4.iloc[0, np.arange(2, 5)] = np.nan
    df5 = df1.copy().astype({4: 'float', 5: 'float', 6: 'float'})
    df5.iloc[0, np.arange(4, 7)] = np.nan
    for left, right in [(df1, df2), (df2, df3), (df4, df5)]:
        res = left + right
        expected = DataFrame({i: left[i] + right[i] for i in left.columns})
        tm.assert_frame_equal(res, expected)