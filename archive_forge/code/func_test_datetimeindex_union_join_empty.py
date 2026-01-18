from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_datetimeindex_union_join_empty(self, sort):
    dti = date_range(start='1/1/2001', end='2/1/2001', freq='D')
    empty = Index([])
    result = dti.union(empty, sort=sort)
    expected = dti.astype('O')
    tm.assert_index_equal(result, expected)
    result = dti.join(empty)
    assert isinstance(result, DatetimeIndex)
    tm.assert_index_equal(result, dti)