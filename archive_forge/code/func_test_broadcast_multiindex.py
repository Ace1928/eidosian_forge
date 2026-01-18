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
@pytest.mark.parametrize('level', [0, None])
def test_broadcast_multiindex(self, level):
    df1 = DataFrame({'A': [0, 1, 2], 'B': [1, 2, 3]})
    df1.columns = df1.columns.set_names('L1')
    df2 = DataFrame({('A', 'C'): [0, 0, 0], ('A', 'D'): [0, 0, 0]})
    df2.columns = df2.columns.set_names(['L1', 'L2'])
    result = df1.add(df2, level=level)
    expected = DataFrame({('A', 'C'): [0, 1, 2], ('A', 'D'): [0, 1, 2]})
    expected.columns = expected.columns.set_names(['L1', 'L2'])
    tm.assert_frame_equal(result, expected)