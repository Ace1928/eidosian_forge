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
def test_frame_sub_nullable_int(any_int_ea_dtype):
    series1 = Series([1, 2, None], dtype=any_int_ea_dtype)
    series2 = Series([1, 2, 3], dtype=any_int_ea_dtype)
    expected = DataFrame([0, 0, None], dtype=any_int_ea_dtype)
    result = series1.to_frame() - series2.to_frame()
    tm.assert_frame_equal(result, expected)