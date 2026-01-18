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
def test_df_add_td64_columnwise(self):
    dti = pd.date_range('2016-01-01', periods=10)
    tdi = pd.timedelta_range('1', periods=10)
    tser = Series(tdi)
    df = DataFrame({0: dti, 1: tdi})
    result = df.add(tser, axis=0)
    expected = DataFrame({0: dti + tdi, 1: tdi + tdi})
    tm.assert_frame_equal(result, expected)