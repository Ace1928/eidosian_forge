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
def test_arith_flex_frame(self, all_arithmetic_operators, float_frame, mixed_float_frame):
    op = all_arithmetic_operators

    def f(x, y):
        if op.startswith('__r'):
            return getattr(operator, op.replace('__r', '__'))(y, x)
        return getattr(operator, op)(x, y)
    result = getattr(float_frame, op)(2 * float_frame)
    expected = f(float_frame, 2 * float_frame)
    tm.assert_frame_equal(result, expected)
    result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
    expected = f(mixed_float_frame, 2 * mixed_float_frame)
    tm.assert_frame_equal(result, expected)
    _check_mixed_float(result, dtype={'C': None})