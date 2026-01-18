import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('f', ['corr', 'cov'])
def test_expanding_corr_cov(self, f, frame):
    g = frame.groupby('A')
    r = g.expanding()
    result = getattr(r, f)(frame)

    def func_0(x):
        return getattr(x.expanding(), f)(frame)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(func_0)
    null_idx = list(range(20, 61)) + list(range(72, 113))
    expected.iloc[null_idx, 1] = np.nan
    expected['A'] = np.nan
    tm.assert_frame_equal(result, expected)
    result = getattr(r.B, f)(pairwise=True)

    def func_1(x):
        return getattr(x.B.expanding(), f)(pairwise=True)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        expected = g.apply(func_1)
    tm.assert_series_equal(result, expected)