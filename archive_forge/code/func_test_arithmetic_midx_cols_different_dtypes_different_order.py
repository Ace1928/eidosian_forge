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
def test_arithmetic_midx_cols_different_dtypes_different_order(self):
    midx = MultiIndex.from_arrays([Series([1, 2]), Series([3, 4])])
    midx2 = MultiIndex.from_arrays([Series([2, 1], dtype='Int8'), Series([4, 3])])
    left = DataFrame([[1, 2], [3, 4]], columns=midx)
    right = DataFrame([[1, 2], [3, 4]], columns=midx2)
    result = left - right
    expected = DataFrame([[-1, 1], [-1, 1]], columns=midx)
    tm.assert_frame_equal(result, expected)