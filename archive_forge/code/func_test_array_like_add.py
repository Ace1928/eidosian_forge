import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_array_like_add(self):

    def check(result):
        _assert_equal_type_and_value(result, ArrayLike(0))
    check(ArrayLike(0) + 0)
    check(0 + ArrayLike(0))
    check(ArrayLike(0) + np.array(0))
    check(np.array(0) + ArrayLike(0))
    check(ArrayLike(np.array(0)) + 0)
    check(0 + ArrayLike(np.array(0)))
    check(ArrayLike(np.array(0)) + np.array(0))
    check(np.array(0) + ArrayLike(np.array(0)))