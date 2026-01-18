import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_inplace(self):
    array_like = ArrayLike(np.array([0]))
    array_like += 1
    _assert_equal_type_and_value(array_like, ArrayLike(np.array([1])))
    array = np.array([0])
    array += ArrayLike(1)
    _assert_equal_type_and_value(array, ArrayLike(np.array([1])))