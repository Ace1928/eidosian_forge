import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_isintlike(self):
    assert_equal(sputils.isintlike(-4), True)
    assert_equal(sputils.isintlike(np.array(3)), True)
    assert_equal(sputils.isintlike(np.array([3])), False)
    with assert_raises(ValueError, match='Inexact indices into sparse matrices are not allowed'):
        sputils.isintlike(3.0)
    assert_equal(sputils.isintlike(2.5), False)
    assert_equal(sputils.isintlike(1 + 3j), False)
    assert_equal(sputils.isintlike((1,)), False)
    assert_equal(sputils.isintlike((1, 2)), False)