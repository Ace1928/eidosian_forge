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
def test_fancy_indexing_empty(self):
    B = asmatrix(arange(50).reshape(5, 10))
    B[1, :] = 0
    B[:, 2] = 0
    B[3, 6] = 0
    A = self.spcreator(B)
    K = np.array([False, False, False, False, False])
    assert_equal(toarray(A[K]), B[K])
    K = np.array([], dtype=int)
    assert_equal(toarray(A[K]), B[K])
    assert_equal(toarray(A[K, K]), B[K, K])
    J = np.array([0, 1, 2, 3, 4], dtype=int)[:, None]
    assert_equal(toarray(A[K, J]), B[K, J])
    assert_equal(toarray(A[J, K]), B[J, K])