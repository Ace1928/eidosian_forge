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
def test_df_add_flex_filled_mixed_dtypes(self):
    dti = pd.date_range('2016-01-01', periods=3)
    ser = Series(['1 Day', 'NaT', '2 Days'], dtype='timedelta64[ns]')
    df = DataFrame({'A': dti, 'B': ser})
    other = DataFrame({'A': ser, 'B': ser})
    fill = pd.Timedelta(days=1).to_timedelta64()
    result = df.add(other, fill_value=fill)
    expected = DataFrame({'A': Series(['2016-01-02', '2016-01-03', '2016-01-05'], dtype='datetime64[ns]'), 'B': ser * 2})
    tm.assert_frame_equal(result, expected)