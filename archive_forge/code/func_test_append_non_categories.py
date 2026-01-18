import pytest
from pandas import (
import pandas._testing as tm
def test_append_non_categories(self, ci):
    result = ci.append(Index(['a', 'd']))
    expected = Index(['a', 'a', 'b', 'b', 'c', 'a', 'a', 'd'])
    tm.assert_index_equal(result, expected, exact=True)