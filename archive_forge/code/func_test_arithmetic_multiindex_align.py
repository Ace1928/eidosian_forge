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
def test_arithmetic_multiindex_align():
    """
    Regression test for: https://github.com/pandas-dev/pandas/issues/33765
    """
    df1 = DataFrame([[1]], index=['a'], columns=MultiIndex.from_product([[0], [1]], names=['a', 'b']))
    df2 = DataFrame([[1]], index=['a'], columns=Index([0], name='a'))
    expected = DataFrame([[0]], index=['a'], columns=MultiIndex.from_product([[0], [1]], names=['a', 'b']))
    result = df1 - df2
    tm.assert_frame_equal(result, expected)