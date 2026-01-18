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
def test_arith_reindex_with_duplicates():
    df1 = DataFrame(data=[[0]], columns=['second'])
    df2 = DataFrame(data=[[0, 0, 0]], columns=['first', 'second', 'second'])
    result = df1 + df2
    expected = DataFrame([[np.nan, 0, 0]], columns=['first', 'second', 'second'])
    tm.assert_frame_equal(result, expected)