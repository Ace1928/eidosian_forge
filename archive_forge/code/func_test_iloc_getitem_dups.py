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
def test_iloc_getitem_dups(self):
    df1 = DataFrame([{'A': None, 'B': 1}, {'A': 2, 'B': 2}])
    df2 = DataFrame([{'A': 3, 'B': 3}, {'A': 4, 'B': 4}])
    df = concat([df1, df2], axis=1)
    result = df.iloc[0, 0]
    assert isna(result)
    result = df.iloc[0, :]
    expected = Series([np.nan, 1, 3, 3], index=['A', 'B', 'A', 'B'], name=0)
    tm.assert_series_equal(result, expected)