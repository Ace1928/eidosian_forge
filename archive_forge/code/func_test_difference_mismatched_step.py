from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_difference_mismatched_step(self):
    obj = RangeIndex.from_range(range(1, 10), name='foo')
    result = obj.difference(obj[::2])
    expected = obj[1::2]
    tm.assert_index_equal(result, expected, exact=True)
    result = obj[::-1].difference(obj[::2], sort=False)
    tm.assert_index_equal(result, expected[::-1], exact=True)
    result = obj.difference(obj[1::2])
    expected = obj[::2]
    tm.assert_index_equal(result, expected, exact=True)
    result = obj[::-1].difference(obj[1::2], sort=False)
    tm.assert_index_equal(result, expected[::-1], exact=True)