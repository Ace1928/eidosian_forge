import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('group_keys', [True, False])
def test_groupby_rolling_group_keys(self, group_keys):
    arrays = [['val1', 'val1', 'val2'], ['val1', 'val1', 'val2']]
    index = MultiIndex.from_arrays(arrays, names=('idx1', 'idx2'))
    s = Series([1, 2, 3], index=index)
    result = s.groupby(['idx1', 'idx2'], group_keys=group_keys).rolling(1).mean()
    expected = Series([1.0, 2.0, 3.0], index=MultiIndex.from_tuples([('val1', 'val1', 'val1', 'val1'), ('val1', 'val1', 'val1', 'val1'), ('val2', 'val2', 'val2', 'val2')], names=['idx1', 'idx2', 'idx1', 'idx2']))
    tm.assert_series_equal(result, expected)