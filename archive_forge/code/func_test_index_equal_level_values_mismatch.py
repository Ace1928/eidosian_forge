import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_equal_level_values_mismatch(check_exact, rtol):
    idx1 = MultiIndex.from_tuples([('A', 2), ('A', 2), ('B', 3), ('B', 4)])
    idx2 = MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 3), ('B', 4)])
    kwargs = {'check_exact': check_exact, 'rtol': rtol}
    msg = "MultiIndex level \\[1\\] are different\n\nMultiIndex level \\[1\\] values are different \\(25\\.0 %\\)\n\\[left\\]:  Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)\n\\[right\\]: Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)