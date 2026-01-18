import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('s1name,s2name', [(np.int64(190), (43, 0)), (190, (43, 0))])
def test_concat_series_name_npscalar_tuple(self, s1name, s2name):
    s1 = Series({'a': 1, 'b': 2}, name=s1name)
    s2 = Series({'c': 5, 'd': 6}, name=s2name)
    result = concat([s1, s2])
    expected = Series({'a': 1, 'b': 2, 'c': 5, 'd': 6})
    tm.assert_series_equal(result, expected)