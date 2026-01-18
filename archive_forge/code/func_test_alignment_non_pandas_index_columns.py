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
def test_alignment_non_pandas_index_columns(self):
    index = ['A', 'B', 'C']
    columns = ['X', 'Y', 'Z']
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), index=index, columns=columns)
    align = DataFrame._align_for_op
    val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tm.assert_frame_equal(align(df, val, axis=0)[1], DataFrame(val, index=df.index, columns=df.columns))
    tm.assert_frame_equal(align(df, val, axis=1)[1], DataFrame(val, index=df.index, columns=df.columns))
    msg = 'Unable to coerce to DataFrame, shape must be'
    val = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match=msg):
        align(df, val, axis=0)
    with pytest.raises(ValueError, match=msg):
        align(df, val, axis=1)
    val = np.zeros((3, 3, 3))
    msg = re.escape('Unable to coerce to Series/DataFrame, dimension must be <= 2: (3, 3, 3)')
    with pytest.raises(ValueError, match=msg):
        align(df, val, axis=0)
    with pytest.raises(ValueError, match=msg):
        align(df, val, axis=1)