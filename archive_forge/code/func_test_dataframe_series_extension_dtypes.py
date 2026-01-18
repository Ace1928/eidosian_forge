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
def test_dataframe_series_extension_dtypes():
    df = DataFrame(np.random.default_rng(2).integers(0, 100, (10, 3)), columns=['a', 'b', 'c'])
    ser = Series([1, 2, 3], index=['a', 'b', 'c'])
    expected = df.to_numpy('int64') + ser.to_numpy('int64').reshape(-1, 3)
    expected = DataFrame(expected, columns=df.columns, dtype='Int64')
    df_ea = df.astype('Int64')
    result = df_ea + ser
    tm.assert_frame_equal(result, expected)
    result = df_ea + ser.astype('Int64')
    tm.assert_frame_equal(result, expected)