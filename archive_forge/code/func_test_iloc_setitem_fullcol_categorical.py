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
@pytest.mark.parametrize('key', [slice(None), slice(3), range(3), [0, 1, 2], Index(range(3)), np.asarray([0, 1, 2])])
@pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
def test_iloc_setitem_fullcol_categorical(self, indexer, key, using_array_manager):
    frame = DataFrame({0: range(3)}, dtype=object)
    cat = Categorical(['alpha', 'beta', 'gamma'])
    if not using_array_manager:
        assert frame._mgr.blocks[0]._can_hold_element(cat)
    df = frame.copy()
    orig_vals = df.values
    indexer(df)[key, 0] = cat
    expected = DataFrame({0: cat}).astype(object)
    if not using_array_manager:
        assert np.shares_memory(df[0].values, orig_vals)
    tm.assert_frame_equal(df, expected)
    df.iloc[0, 0] = 'gamma'
    assert cat[0] != 'gamma'
    frame = DataFrame({0: np.array([0, 1, 2], dtype=object), 1: range(3)})
    df = frame.copy()
    indexer(df)[key, 0] = cat
    expected = DataFrame({0: Series(cat.astype(object), dtype=object), 1: range(3)})
    tm.assert_frame_equal(df, expected)