import pytest
from pandas import (
def test_get_set_value_no_partial_indexing(self):
    index = MultiIndex.from_tuples([(0, 1), (0, 2), (1, 1), (1, 2)])
    df = DataFrame(index=index, columns=range(4))
    with pytest.raises(KeyError, match='^0$'):
        df._get_value(0, 1)