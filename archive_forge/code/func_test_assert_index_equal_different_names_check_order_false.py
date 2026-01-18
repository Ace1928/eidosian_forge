import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_index_equal_different_names_check_order_false():
    idx1 = Index([1, 3], name='a')
    idx2 = Index([3, 1], name='b')
    with pytest.raises(AssertionError, match='"names" are different'):
        tm.assert_index_equal(idx1, idx2, check_order=False, check_names=True)