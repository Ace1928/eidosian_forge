import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_getitem_partial_int(self):
    l1 = [10, 20]
    l2 = ['a', 'b']
    df = DataFrame(index=range(2), columns=MultiIndex.from_product([l1, l2]))
    expected = DataFrame(index=range(2), columns=l2)
    result = df[20]
    tm.assert_frame_equal(result, expected)
    expected = DataFrame(index=range(2), columns=MultiIndex.from_product([l1[1:], l2]))
    result = df[[20]]
    tm.assert_frame_equal(result, expected)
    with pytest.raises(KeyError, match='1'):
        df[1]
    with pytest.raises(KeyError, match="'\\[1\\] not in index'"):
        df[[1]]