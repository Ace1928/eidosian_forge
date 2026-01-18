import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_unicode():
    msg = 'numpy array are different\n\nnumpy array values are different \\(33\\.33333 %\\)\n\\[left\\]:  \\[á, à, ä\\]\n\\[right\\]: \\[á, à, å\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(np.array(['á', 'à', 'ä']), np.array(['á', 'à', 'å']))