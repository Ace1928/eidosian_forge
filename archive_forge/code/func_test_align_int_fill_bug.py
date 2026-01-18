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
def test_align_int_fill_bug(self):
    X = np.arange(10 * 10, dtype='float64').reshape(10, 10)
    Y = np.ones((10, 1), dtype=int)
    df1 = DataFrame(X)
    df1['0.X'] = Y.squeeze()
    df2 = df1.astype(float)
    result = df1 - df1.mean()
    expected = df2 - df2.mean()
    tm.assert_frame_equal(result, expected)