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
def test_iloc_setitem_pandas_object(self):
    s_orig = Series([0, 1, 2, 3])
    expected = Series([0, -1, -2, 3])
    s = s_orig.copy()
    s.iloc[Series([1, 2])] = [-1, -2]
    tm.assert_series_equal(s, expected)
    s = s_orig.copy()
    s.iloc[Index([1, 2])] = [-1, -2]
    tm.assert_series_equal(s, expected)