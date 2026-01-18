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
def test_df_boolean_comparison_error(self):
    df = DataFrame(np.arange(6).reshape((3, 2)))
    expected = DataFrame([[False, False], [True, False], [False, False]])
    result = df == (2, 2)
    tm.assert_frame_equal(result, expected)
    result = df == [2, 2]
    tm.assert_frame_equal(result, expected)