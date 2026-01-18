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
def test_frame_multiindex_operations_no_align(self):
    df = DataFrame({2010: [1, 2, 3], 2020: [3, 4, 5]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
    series = Series([0.4], index=MultiIndex.from_product([['c'], ['a']], names=['mod', 'scen']))
    expected = DataFrame({2010: np.nan, 2020: np.nan}, index=MultiIndex.from_tuples([('a', 'b', 0), ('a', 'b', 1), ('a', 'b', 2), ('a', 'c', np.nan)], names=['scen', 'mod', 'id']))
    result = df.add(series, axis=0)
    tm.assert_frame_equal(result, expected)