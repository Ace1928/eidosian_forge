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
def test_add_with_dti_mismatched_tzs(self):
    base = pd.DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], tz='UTC')
    idx1 = base.tz_convert('Asia/Tokyo')[:2]
    idx2 = base.tz_convert('US/Eastern')[1:]
    df1 = DataFrame({'A': [1, 2]}, index=idx1)
    df2 = DataFrame({'A': [1, 1]}, index=idx2)
    exp = DataFrame({'A': [np.nan, 3, np.nan]}, index=base)
    tm.assert_frame_equal(df1 + df2, exp)