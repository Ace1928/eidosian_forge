from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def test_adjoint_conjugate():
    X = np.array([[1j]])
    A = interface.aslinearoperator(X)
    B = 1j * A
    Y = 1j * X
    v = np.array([1])
    assert_equal(B.dot(v), Y.dot(v))
    assert_equal(B.H.dot(v), Y.T.conj().dot(v))