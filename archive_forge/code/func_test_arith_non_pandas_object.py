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
def test_arith_non_pandas_object(self):
    df = DataFrame(np.arange(1, 10, dtype='f8').reshape(3, 3), columns=['one', 'two', 'three'], index=['a', 'b', 'c'])
    val1 = df.xs('a').values
    added = DataFrame(df.values + val1, index=df.index, columns=df.columns)
    tm.assert_frame_equal(df + val1, added)
    added = DataFrame((df.values.T + val1).T, index=df.index, columns=df.columns)
    tm.assert_frame_equal(df.add(val1, axis=0), added)
    val2 = list(df['two'])
    added = DataFrame(df.values + val2, index=df.index, columns=df.columns)
    tm.assert_frame_equal(df + val2, added)
    added = DataFrame((df.values.T + val2).T, index=df.index, columns=df.columns)
    tm.assert_frame_equal(df.add(val2, axis='index'), added)
    val3 = np.random.default_rng(2).random(df.shape)
    added = DataFrame(df.values + val3, index=df.index, columns=df.columns)
    tm.assert_frame_equal(df.add(val3), added)