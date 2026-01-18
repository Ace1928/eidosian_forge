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
def test_iloc_getitem_with_duplicates2(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1, 1, 2])
    result = df.iloc[:, [0]]
    expected = df.take([0], axis=1)
    tm.assert_frame_equal(result, expected)