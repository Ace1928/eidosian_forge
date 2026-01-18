import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
def test_groupby_rolling_no_sort(self):
    result = DataFrame({'foo': [2, 1], 'bar': [2, 1]}).groupby('foo', sort=False).rolling(1).min()
    expected = DataFrame(np.array([[2.0, 2.0], [1.0, 1.0]]), columns=['foo', 'bar'], index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=['foo', None]))
    expected = expected.drop(columns='foo')
    tm.assert_frame_equal(result, expected)