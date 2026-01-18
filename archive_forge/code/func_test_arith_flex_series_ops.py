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
@pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'mod'])
def test_arith_flex_series_ops(self, simple_frame, op):
    df = simple_frame
    row = df.xs('a')
    col = df['two']
    f = getattr(df, op)
    op = getattr(operator, op)
    tm.assert_frame_equal(f(row), op(df, row))
    tm.assert_frame_equal(f(col, axis=0), op(df.T, col).T)