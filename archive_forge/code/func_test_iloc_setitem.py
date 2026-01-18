from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_setitem(self, warn_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=np.arange(0, 8, 2), columns=np.arange(0, 12, 3))
    df.iloc[1, 1] = 1
    result = df.iloc[1, 1]
    assert result == 1
    df.iloc[:, 2:3] = 0
    expected = df.iloc[:, 2:3]
    result = df.iloc[:, 2:3]
    tm.assert_frame_equal(result, expected)
    s = Series(0, index=[4, 5, 6])
    s.iloc[1:2] += 1
    expected = Series([0, 1, 0], index=[4, 5, 6])
    tm.assert_series_equal(s, expected)