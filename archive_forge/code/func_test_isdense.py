import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_isdense(self):
    assert_equal(sputils.isdense(np.array([1])), True)
    assert_equal(sputils.isdense(matrix([1])), True)