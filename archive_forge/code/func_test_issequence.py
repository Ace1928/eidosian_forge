import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_issequence(self):
    assert_equal(sputils.issequence((1,)), True)
    assert_equal(sputils.issequence((1, 2, 3)), True)
    assert_equal(sputils.issequence([1]), True)
    assert_equal(sputils.issequence([1, 2, 3]), True)
    assert_equal(sputils.issequence(np.array([1, 2, 3])), True)
    assert_equal(sputils.issequence(np.array([[1], [2], [3]])), False)
    assert_equal(sputils.issequence(3), False)