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
def test_iloc_setitem_list(self):
    df = DataFrame(np.arange(9).reshape((3, 3)), index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
    df.iloc[[0, 1], [1, 2]]
    df.iloc[[0, 1], [1, 2]] += 100
    expected = DataFrame(np.array([0, 101, 102, 3, 104, 105, 6, 7, 8]).reshape((3, 3)), index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(df, expected)