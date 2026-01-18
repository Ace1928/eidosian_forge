import pytest
from pandas import (
import pandas._testing as tm
def test_map_bug_1677(self):
    index = DatetimeIndex(['2012-04-25 09:30:00.393000'])
    f = index.asof
    result = index.map(f)
    expected = Index([f(index[0])])
    tm.assert_index_equal(result, expected)