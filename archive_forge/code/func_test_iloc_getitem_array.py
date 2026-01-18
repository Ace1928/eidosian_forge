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
def test_iloc_getitem_array(self):
    df = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}, {'A': 1000, 'B': 2000, 'C': 3000}])
    expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}])
    tm.assert_frame_equal(df.iloc[[0]], expected)
    expected = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 100, 'B': 200, 'C': 300}])
    tm.assert_frame_equal(df.iloc[[0, 1]], expected)
    expected = DataFrame([{'B': 2, 'C': 3}, {'B': 2000, 'C': 3000}], index=[0, 2])
    result = df.iloc[[0, 2], [1, 2]]
    tm.assert_frame_equal(result, expected)