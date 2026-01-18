import copy
import numpy as np
import pytest
import pandas as pd
from pandas import Timestamp
import pandas._testing as tm
@pytest.mark.parametrize('a,b,klass1,klass2', [(np.array([1]), 1, 'ndarray', 'int'), (1, np.array([1]), 'int', 'ndarray')])
def test_assert_numpy_array_equal_class_mismatch(a, b, klass1, klass2):
    msg = f'numpy array are different\n\nnumpy array classes are different\n\\[left\\]:  {klass1}\n\\[right\\]: {klass2}'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_numpy_array_equal(a, b)