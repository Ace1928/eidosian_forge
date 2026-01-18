from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_difference_endpoints_overlap_interior_preserved(self):
    left = RangeIndex(-8, 20, 7)
    right = RangeIndex(13, -9, -3)
    result = left.difference(right)
    expected = RangeIndex(-1, 13, 7)
    assert expected.tolist() == [-1, 6]
    tm.assert_index_equal(result, expected, exact=True)