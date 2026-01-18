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
def test_flex_add_scalar_fill_value(self):
    dat = np.array([0, 1, np.nan, 3, 4, 5], dtype='float')
    df = DataFrame({'foo': dat}, index=range(6))
    exp = df.fillna(0).add(2)
    res = df.add(2, fill_value=0)
    tm.assert_frame_equal(res, exp)