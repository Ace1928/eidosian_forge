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
def test_iloc_setitem_axis_argument(self):
    df = DataFrame([[6, 'c', 10], [7, 'd', 11], [8, 'e', 12]])
    df[1] = df[1].astype(object)
    expected = DataFrame([[6, 'c', 10], [7, 'd', 11], [5, 5, 5]])
    expected[1] = expected[1].astype(object)
    df.iloc(axis=0)[2] = 5
    tm.assert_frame_equal(df, expected)
    df = DataFrame([[6, 'c', 10], [7, 'd', 11], [8, 'e', 12]])
    df[1] = df[1].astype(object)
    expected = DataFrame([[6, 'c', 5], [7, 'd', 5], [8, 'e', 5]])
    expected[1] = expected[1].astype(object)
    df.iloc(axis=1)[2] = 5
    tm.assert_frame_equal(df, expected)