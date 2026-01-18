from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('box', [array, Series])
def test_iloc_setitem_ea_inplace(self, frame_or_series, box, using_copy_on_write):
    arr = array([1, 2, 3, 4])
    obj = frame_or_series(arr.to_numpy('i8'))
    if frame_or_series is Series:
        values = obj.values
    else:
        values = obj._mgr.arrays[0]
    if frame_or_series is Series:
        obj.iloc[:2] = box(arr[2:])
    else:
        obj.iloc[:2, 0] = box(arr[2:])
    expected = frame_or_series(np.array([3, 4, 3, 4], dtype='i8'))
    tm.assert_equal(obj, expected)
    if frame_or_series is Series:
        if using_copy_on_write:
            assert obj.values is not values
            assert np.shares_memory(obj.values, values)
        else:
            assert obj.values is values
    else:
        assert np.shares_memory(obj[0].values, values)