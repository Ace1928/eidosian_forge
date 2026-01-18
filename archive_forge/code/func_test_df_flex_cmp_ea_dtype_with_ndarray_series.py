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
def test_df_flex_cmp_ea_dtype_with_ndarray_series(self):
    ii = pd.IntervalIndex.from_breaks([1, 2, 3])
    df = DataFrame({'A': ii, 'B': ii})
    ser = Series([0, 0])
    res = df.eq(ser, axis=0)
    expected = DataFrame({'A': [False, False], 'B': [False, False]})
    tm.assert_frame_equal(res, expected)
    ser2 = Series([1, 2], index=['A', 'B'])
    res2 = df.eq(ser2, axis=1)
    tm.assert_frame_equal(res2, expected)