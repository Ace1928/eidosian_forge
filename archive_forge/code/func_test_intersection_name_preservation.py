from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('index2,keeps_name', [(Index([3, 4, 5, 6, 7], name='index'), True), (Index([3, 4, 5, 6, 7], name='other'), False), (Index([3, 4, 5, 6, 7]), False)])
def test_intersection_name_preservation(self, index2, keeps_name, sort):
    index1 = Index([1, 2, 3, 4, 5], name='index')
    expected = Index([3, 4, 5])
    result = index1.intersection(index2, sort)
    if keeps_name:
        expected.name = 'index'
    assert result.name == expected.name
    tm.assert_index_equal(result, expected)