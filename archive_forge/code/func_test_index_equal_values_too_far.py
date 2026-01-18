import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_values_too_far(check_exact, rtol):
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 4])
    kwargs = {'check_exact': check_exact, 'rtol': rtol}
    msg = "Index are different\n\nIndex values are different \\(33\\.33333 %\\)\n\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)\n\\[right\\]: Index\\(\\[1, 2, 4\\], dtype='int64'\\)"
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)