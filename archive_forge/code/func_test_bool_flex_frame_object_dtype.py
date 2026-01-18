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
def test_bool_flex_frame_object_dtype(self):
    df1 = DataFrame({'col': ['foo', np.nan, 'bar']}, dtype=object)
    df2 = DataFrame({'col': ['foo', datetime.now(), 'bar']}, dtype=object)
    result = df1.ne(df2)
    exp = DataFrame({'col': [False, True, False]})
    tm.assert_frame_equal(result, exp)