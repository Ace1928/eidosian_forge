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
def test_arith_flex_frame_corner(self, float_frame):
    const_add = float_frame.add(1)
    tm.assert_frame_equal(const_add, float_frame + 1)
    result = float_frame.add(float_frame[:0])
    expected = float_frame.sort_index() * np.nan
    tm.assert_frame_equal(result, expected)
    result = float_frame[:0].add(float_frame)
    expected = float_frame.sort_index() * np.nan
    tm.assert_frame_equal(result, expected)
    with pytest.raises(NotImplementedError, match='fill_value'):
        float_frame.add(float_frame.iloc[0], fill_value=3)
    with pytest.raises(NotImplementedError, match='fill_value'):
        float_frame.add(float_frame.iloc[0], axis='index', fill_value=3)