import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('f', ['std', 'var'])
def test_expanding_ddof(self, f, frame):
    g = frame.groupby('A', group_keys=False)
    r = g.expanding()
    result = getattr(r, f)(ddof=0)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
    expected = expected.drop('A', axis=1)
    expected_index = MultiIndex.from_arrays([frame['A'], range(40)])
    expected.index = expected_index
    tm.assert_frame_equal(result, expected)