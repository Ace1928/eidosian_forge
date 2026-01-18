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
def test_comparison_with_categorical_dtype(self):
    df = DataFrame({'A': ['foo', 'bar', 'baz']})
    exp = DataFrame({'A': [True, False, False]})
    res = df == 'foo'
    tm.assert_frame_equal(res, exp)
    df['A'] = df['A'].astype('category')
    res = df == 'foo'
    tm.assert_frame_equal(res, exp)