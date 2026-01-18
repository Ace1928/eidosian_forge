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
def test_iloc_assign_series_to_df_cell(self):
    df = DataFrame(columns=['a'], index=[0])
    df.iloc[0, 0] = Series([1, 2, 3])
    expected = DataFrame({'a': [Series([1, 2, 3])]}, columns=['a'], index=[0])
    tm.assert_frame_equal(df, expected)