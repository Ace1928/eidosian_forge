import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_matrix_properties(self):
    a = np.matrix([1.0], dtype=float)
    assert_(type(a.real) is np.matrix)
    assert_(type(a.imag) is np.matrix)
    c, d = np.matrix([0.0]).nonzero()
    assert_(type(c) is np.ndarray)
    assert_(type(d) is np.ndarray)