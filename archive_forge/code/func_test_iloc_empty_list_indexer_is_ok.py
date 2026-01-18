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
def test_iloc_empty_list_indexer_is_ok(self):
    df = DataFrame(np.ones((5, 2)), index=Index([f'i-{i}' for i in range(5)], name='a'), columns=Index([f'i-{i}' for i in range(2)], name='a'))
    tm.assert_frame_equal(df.iloc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True)
    tm.assert_frame_equal(df.iloc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True)
    tm.assert_frame_equal(df.iloc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True)