import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_fancy_indexing_sparse_boolean(self):
    np.random.seed(1234)
    B = asmatrix(arange(50).reshape(5, 10))
    A = self.spcreator(B)
    X = np.array(np.random.randint(0, 2, size=(5, 10)), dtype=bool)
    Xsp = csr_matrix(X)
    assert_equal(toarray(A[Xsp]), B[X])
    assert_equal(toarray(A[A > 9]), B[B > 9])
    Z = np.array(np.random.randint(0, 2, size=(5, 11)), dtype=bool)
    Y = np.array(np.random.randint(0, 2, size=(6, 10)), dtype=bool)
    Zsp = csr_matrix(Z)
    Ysp = csr_matrix(Y)
    assert_raises(IndexError, A.__getitem__, Zsp)
    assert_raises(IndexError, A.__getitem__, Ysp)
    assert_raises((IndexError, ValueError), A.__getitem__, (Xsp, 1))