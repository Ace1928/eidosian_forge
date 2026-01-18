import pytest
from pandas import (
import pandas._testing as tm
def test_append_category_objects(self, ci):
    result = ci.append(Index(['c', 'a']))
    expected = CategoricalIndex(list('aabbcaca'), categories=ci.categories)
    tm.assert_index_equal(result, expected, exact=True)