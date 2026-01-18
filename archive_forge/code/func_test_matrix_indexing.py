import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
from numpy.ma.extras import mr_
from numpy.compat import pickle
def test_matrix_indexing(self):
    x1 = np.matrix([[1, 2, 3], [4, 3, 2]])
    x2 = masked_array(x1, mask=[[1, 0, 0], [0, 1, 0]])
    x3 = masked_array(x1, mask=[[0, 1, 0], [1, 0, 0]])
    x4 = masked_array(x1)
    str(x2)
    repr(x2)
    assert_(type(x2[1, 0]) is type(x1[1, 0]))
    assert_(x1[1, 0] == x2[1, 0])
    assert_(x2[1, 1] is masked)
    assert_equal(x1[0, 2], x2[0, 2])
    assert_equal(x1[0, 1:], x2[0, 1:])
    assert_equal(x1[:, 2], x2[:, 2])
    assert_equal(x1[:], x2[:])
    assert_equal(x1[1:], x3[1:])
    x1[0, 2] = 9
    x2[0, 2] = 9
    assert_equal(x1, x2)
    x1[0, 1:] = 99
    x2[0, 1:] = 99
    assert_equal(x1, x2)
    x2[0, 1] = masked
    assert_equal(x1, x2)
    x2[0, 1:] = masked
    assert_equal(x1, x2)
    x2[0, :] = x1[0, :]
    x2[0, 1] = masked
    assert_(allequal(getmask(x2), np.array([[0, 1, 0], [0, 1, 0]])))
    x3[1, :] = masked_array([1, 2, 3], [1, 1, 0])
    assert_(allequal(getmask(x3)[1], masked_array([1, 1, 0])))
    assert_(allequal(getmask(x3[1]), masked_array([1, 1, 0])))
    x4[1, :] = masked_array([1, 2, 3], [1, 1, 0])
    assert_(allequal(getmask(x4[1]), masked_array([1, 1, 0])))
    assert_(allequal(x4[1], masked_array([1, 2, 3])))
    x1 = np.matrix(np.arange(5) * 1.0)
    x2 = masked_values(x1, 3.0)
    assert_equal(x1, x2)
    assert_(allequal(masked_array([0, 0, 0, 1, 0], dtype=MaskType), x2.mask))
    assert_equal(3.0, x2.fill_value)