import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_iterable_length_mismatch():
    msg = 'Iterable are different\n\nIterable length are different\n\\[left\\]:  2\n\\[right\\]: 3'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal([1, 2], [3, 4, 5])