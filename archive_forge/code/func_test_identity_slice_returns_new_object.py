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
def test_identity_slice_returns_new_object(self, using_copy_on_write, warn_copy_on_write):
    original_df = DataFrame({'a': [1, 2, 3]})
    sliced_df = original_df.iloc[:]
    assert sliced_df is not original_df
    assert np.shares_memory(original_df['a'], sliced_df['a'])
    with tm.assert_cow_warning(warn_copy_on_write):
        original_df.loc[:, 'a'] = [4, 4, 4]
    if using_copy_on_write:
        assert (sliced_df['a'] == [1, 2, 3]).all()
    else:
        assert (sliced_df['a'] == 4).all()
    original_series = Series([1, 2, 3, 4, 5, 6])
    sliced_series = original_series.iloc[:]
    assert sliced_series is not original_series
    with tm.assert_cow_warning(warn_copy_on_write):
        original_series[:3] = [7, 8, 9]
    if using_copy_on_write:
        assert all(sliced_series[:3] == [1, 2, 3])
    else:
        assert all(sliced_series[:3] == [7, 8, 9])