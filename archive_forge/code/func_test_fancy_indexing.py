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
def test_fancy_indexing(self):
    B = asmatrix(arange(50).reshape(5, 10))
    A = self.spcreator(B)
    assert_equal(A[[1, 3]].toarray(), B[[1, 3]])
    assert_equal(A[3, [1, 3]].toarray(), B[3, [1, 3]])
    assert_equal(A[-1, [2, -5]].toarray(), B[-1, [2, -5]])
    assert_equal(A[array(-1), [2, -5]].toarray(), B[-1, [2, -5]])
    assert_equal(A[-1, array([2, -5])].toarray(), B[-1, [2, -5]])
    assert_equal(A[array(-1), array([2, -5])].toarray(), B[-1, [2, -5]])
    assert_equal(A[:, [2, 8, 3, -1]].toarray(), B[:, [2, 8, 3, -1]])
    assert_equal(A[3:4, [9]].toarray(), B[3:4, [9]])
    assert_equal(A[1:4, [-1, -5]].toarray(), B[1:4, [-1, -5]])
    assert_equal(A[1:4, array([-1, -5])].toarray(), B[1:4, [-1, -5]])
    assert_equal(A[[1, 3], 3].toarray(), B[[1, 3], 3])
    assert_equal(A[[2, -5], -4].toarray(), B[[2, -5], -4])
    assert_equal(A[array([2, -5]), -4].toarray(), B[[2, -5], -4])
    assert_equal(A[[2, -5], array(-4)].toarray(), B[[2, -5], -4])
    assert_equal(A[array([2, -5]), array(-4)].toarray(), B[[2, -5], -4])
    assert_equal(A[[1, 3], :].toarray(), B[[1, 3], :])
    assert_equal(A[[2, -5], 8:-1].toarray(), B[[2, -5], 8:-1])
    assert_equal(A[array([2, -5]), 8:-1].toarray(), B[[2, -5], 8:-1])
    assert_equal(toarray(A[[1, 3], [2, 4]]), B[[1, 3], [2, 4]])
    assert_equal(toarray(A[[-1, -3], [2, -4]]), B[[-1, -3], [2, -4]])
    assert_equal(toarray(A[array([-1, -3]), [2, -4]]), B[[-1, -3], [2, -4]])
    assert_equal(toarray(A[[-1, -3], array([2, -4])]), B[[-1, -3], [2, -4]])
    assert_equal(toarray(A[array([-1, -3]), array([2, -4])]), B[[-1, -3], [2, -4]])
    assert_equal(A[[[1], [3]], [2, 4]].toarray(), B[[[1], [3]], [2, 4]])
    assert_equal(A[[[-1], [-3], [-2]], [2, -4]].toarray(), B[[[-1], [-3], [-2]], [2, -4]])
    assert_equal(A[array([[-1], [-3], [-2]]), [2, -4]].toarray(), B[[[-1], [-3], [-2]], [2, -4]])
    assert_equal(A[[[-1], [-3], [-2]], array([2, -4])].toarray(), B[[[-1], [-3], [-2]], [2, -4]])
    assert_equal(A[array([[-1], [-3], [-2]]), array([2, -4])].toarray(), B[[[-1], [-3], [-2]], [2, -4]])
    assert_equal(A[[1, 3]].toarray(), B[[1, 3]])
    assert_equal(A[[-1, -3]].toarray(), B[[-1, -3]])
    assert_equal(A[array([-1, -3])].toarray(), B[[-1, -3]])
    assert_equal(A[[1, 3], :][:, [2, 4]].toarray(), B[[1, 3], :][:, [2, 4]])
    assert_equal(A[[-1, -3], :][:, [2, -4]].toarray(), B[[-1, -3], :][:, [2, -4]])
    assert_equal(A[array([-1, -3]), :][:, array([2, -4])].toarray(), B[[-1, -3], :][:, [2, -4]])
    assert_equal(A[:, [1, 3]][[2, 4], :].toarray(), B[:, [1, 3]][[2, 4], :])
    assert_equal(A[:, [-1, -3]][[2, -4], :].toarray(), B[:, [-1, -3]][[2, -4], :])
    assert_equal(A[:, array([-1, -3])][array([2, -4]), :].toarray(), B[:, [-1, -3]][[2, -4], :])
    s = slice(int8(2), int8(4), None)
    assert_equal(A[s, :].toarray(), B[2:4, :])
    assert_equal(A[:, s].toarray(), B[:, 2:4])
    i = np.array([[1]], dtype=int)
    assert_equal(A[i, i].toarray(), B[i, i])
    assert_equal(A[[[]], [[]]].toarray(), B[[[]], [[]]])