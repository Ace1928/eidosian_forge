import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'int32'])
def test_is_range_indexer_not_equal_shape(self, dtype):
    left = np.array([0, 1, 2], dtype=dtype)
    assert not lib.is_range_indexer(left, 2)