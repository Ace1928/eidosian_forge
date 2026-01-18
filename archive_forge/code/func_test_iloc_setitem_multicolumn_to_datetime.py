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
def test_iloc_setitem_multicolumn_to_datetime(self):
    df = DataFrame({'A': ['2022-01-01', '2022-01-02'], 'B': ['2021', '2022']})
    df.iloc[:, [0]] = DataFrame({'A': to_datetime(['2021', '2022'])})
    expected = DataFrame({'A': [Timestamp('2021-01-01 00:00:00'), Timestamp('2022-01-01 00:00:00')], 'B': ['2021', '2022']})
    tm.assert_frame_equal(df, expected, check_dtype=False)