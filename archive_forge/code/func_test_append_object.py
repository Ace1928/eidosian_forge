import pytest
from pandas import (
import pandas._testing as tm
def test_append_object(self, ci):
    result = Index(['c', 'a']).append(ci)
    expected = Index(list('caaabbca'))
    tm.assert_index_equal(result, expected, exact=True)