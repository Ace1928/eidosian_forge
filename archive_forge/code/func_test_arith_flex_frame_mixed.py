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
@pytest.mark.parametrize('op', ['__add__', '__sub__', '__mul__'])
def test_arith_flex_frame_mixed(self, op, int_frame, mixed_int_frame, mixed_float_frame, switch_numexpr_min_elements):
    f = getattr(operator, op)
    result = getattr(mixed_int_frame, op)(2 + mixed_int_frame)
    expected = f(mixed_int_frame, 2 + mixed_int_frame)
    dtype = None
    if op in ['__sub__']:
        dtype = {'B': 'uint64', 'C': None}
    elif op in ['__add__', '__mul__']:
        dtype = {'C': None}
    if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
        dtype['A'] = (2 + mixed_int_frame)['A'].dtype
    tm.assert_frame_equal(result, expected)
    _check_mixed_int(result, dtype=dtype)
    result = getattr(mixed_float_frame, op)(2 * mixed_float_frame)
    expected = f(mixed_float_frame, 2 * mixed_float_frame)
    tm.assert_frame_equal(result, expected)
    _check_mixed_float(result, dtype={'C': None})
    result = getattr(int_frame, op)(2 * int_frame)
    expected = f(int_frame, 2 * int_frame)
    tm.assert_frame_equal(result, expected)