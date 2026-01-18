import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_order', [True, False])
def test_index_equal_value_order_mismatch(check_exact, rtol, check_order):
    idx1 = Index([1, 2, 3])
    idx2 = Index([3, 2, 1])
    msg = "Index are different\n\nIndex values are different \\(66\\.66667 %\\)\n\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: Index\\(\\[3, 2, 1\\], dtype='int64'\\)"
    if check_order:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=True)
    else:
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=False)