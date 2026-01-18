import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_getdtype(self):
    A = np.array([1], dtype='int8')
    assert_equal(sputils.getdtype(None, default=float), float)
    assert_equal(sputils.getdtype(None, a=A), np.int8)
    with assert_raises(ValueError, match='object dtype is not supported by sparse matrices'):
        sputils.getdtype('O')