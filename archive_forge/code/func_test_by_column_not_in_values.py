import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('columns', [MultiIndex.from_tuples([('A', ''), ('B', 'C')]), ['A', 'B']])
def test_by_column_not_in_values(self, columns):
    df = DataFrame([[1, 0]] * 20 + [[2, 0]] * 12 + [[3, 0]] * 8, columns=columns)
    g = df.groupby('A')
    original_obj = g.obj.copy(deep=True)
    r = g.rolling(4)
    result = r.sum()
    assert 'A' not in result.columns
    tm.assert_frame_equal(g.obj, original_obj)