import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_ismatrix(self):
    assert_equal(sputils.ismatrix(((),)), True)
    assert_equal(sputils.ismatrix([[1], [2]]), True)
    assert_equal(sputils.ismatrix(np.arange(3)[None]), True)
    assert_equal(sputils.ismatrix([1, 2]), False)
    assert_equal(sputils.ismatrix(np.arange(3)), False)
    assert_equal(sputils.ismatrix([[[1]]]), False)
    assert_equal(sputils.ismatrix(3), False)