import numpy as np
import pytest
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('val', [(1, 2), np.datetime64('2019-12-31'), np.timedelta64(1, 'D')])
@pytest.mark.parametrize('loc', [-1, 2])
def test_insert_datetime_into_object(self, loc, val):
    idx = Index(['1', '2', '3'])
    result = idx.insert(loc, val)
    expected = Index(['1', '2', val, '3'])
    tm.assert_index_equal(result, expected)
    assert type(expected[2]) is type(val)