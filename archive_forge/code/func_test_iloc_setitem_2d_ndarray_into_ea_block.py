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
def test_iloc_setitem_2d_ndarray_into_ea_block(self):
    df = DataFrame({'status': ['a', 'b', 'c']}, dtype='category')
    df.iloc[np.array([0, 1]), np.array([0])] = np.array([['a'], ['a']])
    expected = DataFrame({'status': ['a', 'a', 'c']}, dtype=df['status'].dtype)
    tm.assert_frame_equal(df, expected)