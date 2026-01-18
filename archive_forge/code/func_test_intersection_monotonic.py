from datetime import (
import numpy as np
import pytest
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('index2,keeps_name', [(Index([4, 7, 6, 5, 3], name='index'), True), (Index([4, 7, 6, 5, 3], name='other'), False)])
def test_intersection_monotonic(self, index2, keeps_name, sort):
    index1 = Index([5, 3, 2, 4, 1], name='index')
    expected = Index([5, 3, 4])
    if keeps_name:
        expected.name = 'index'
    result = index1.intersection(index2, sort=sort)
    if sort is None:
        expected = expected.sort_values()
    tm.assert_index_equal(result, expected)