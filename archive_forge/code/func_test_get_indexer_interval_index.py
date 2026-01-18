import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [IntervalIndex, array, list])
def test_get_indexer_interval_index(self, box):
    rng = period_range('2022-07-01', freq='D', periods=3)
    idx = box(interval_range(Timestamp('2022-07-01'), freq='3D', periods=3))
    actual = rng.get_indexer(idx)
    expected = np.array([-1, -1, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(actual, expected)