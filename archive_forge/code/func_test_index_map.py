import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name', [None, 'name'])
def test_index_map(self, name):
    count = 6
    index = date_range('2018-01-01', periods=count, freq='M', name=name).map(lambda x: (x.year, x.month))
    exp_index = MultiIndex.from_product(((2018,), range(1, 7)), names=[name, name])
    tm.assert_index_equal(index, exp_index)