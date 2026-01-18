from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_col(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 10)), columns=range(0, 20, 2))
    result = df.iloc[:, 1]
    exp = df.loc[:, 2]
    tm.assert_series_equal(result, exp)
    result = df.iloc[:, 2]
    exp = df.loc[:, 4]
    tm.assert_series_equal(result, exp)
    result = df.iloc[:, slice(4, 8)]
    expected = df.loc[:, 8:14]
    tm.assert_frame_equal(result, expected)
    result = df.iloc[:, [1, 2, 4, 6]]
    expected = df.reindex(columns=df.columns[[1, 2, 4, 6]])
    tm.assert_frame_equal(result, expected)