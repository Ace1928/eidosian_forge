import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_isscalarlike(self):
    assert_equal(sputils.isscalarlike(3.0), True)
    assert_equal(sputils.isscalarlike(-4), True)
    assert_equal(sputils.isscalarlike(2.5), True)
    assert_equal(sputils.isscalarlike(1 + 3j), True)
    assert_equal(sputils.isscalarlike(np.array(3)), True)
    assert_equal(sputils.isscalarlike('16'), True)
    assert_equal(sputils.isscalarlike(np.array([3])), False)
    assert_equal(sputils.isscalarlike([[3]]), False)
    assert_equal(sputils.isscalarlike((1,)), False)
    assert_equal(sputils.isscalarlike((1, 2)), False)