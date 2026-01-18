from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def test_identity():
    ident = interface.IdentityOperator((3, 3))
    assert_equal(ident * [1, 2, 3], [1, 2, 3])
    assert_equal(ident.dot(np.arange(9).reshape(3, 3)).ravel(), np.arange(9))
    assert_raises(ValueError, ident.matvec, [1, 2, 3, 4])