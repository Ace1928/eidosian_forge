import pytest
from pandas import (
import pandas._testing as tm
def test_append_to_another(self):
    fst = Index(['a', 'b'])
    snd = CategoricalIndex(['d', 'e'])
    result = fst.append(snd)
    expected = Index(['a', 'b', 'd', 'e'])
    tm.assert_index_equal(result, expected)