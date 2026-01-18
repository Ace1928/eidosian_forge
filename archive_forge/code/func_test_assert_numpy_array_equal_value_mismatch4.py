import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
def test_assert_numpy_array_equal_value_mismatch4():
    msg = 'numpy array are different\n\nnumpy array values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[1\\.1, 2\\.000001\\]\n\\[right\\]: \\[1\\.1, 2.0\\]'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(np.array([1.1, 2.000001]), np.array([1.1, 2.0]))