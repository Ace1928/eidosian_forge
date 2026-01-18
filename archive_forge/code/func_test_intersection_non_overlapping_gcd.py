from datetime import (
from hypothesis import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_intersection_non_overlapping_gcd(self, sort, names):
    index = RangeIndex(1, 10, 2, name=names[0])
    other = RangeIndex(0, 10, 4, name=names[1])
    result = index.intersection(other, sort=sort)
    expected = RangeIndex(0, 0, 1, name=names[2])
    tm.assert_index_equal(result, expected)