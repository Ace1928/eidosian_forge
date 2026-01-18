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
def test_has_canonical_format(self):
    """Ensure has_canonical_format memoizes state for sum_duplicates"""
    M = csr_matrix((np.array([2]), np.array([0]), np.array([0, 1])))
    assert_equal(True, M.has_canonical_format)
    indices = np.array([0, 0])
    data = np.array([1, 1])
    indptr = np.array([0, 2])
    M = csr_matrix((data, indices, indptr)).copy()
    assert_equal(False, M.has_canonical_format)
    assert isinstance(M.has_canonical_format, bool)
    M.sum_duplicates()
    assert_equal(True, M.has_canonical_format)
    assert_equal(1, len(M.indices))
    M = csr_matrix((data, indices, indptr)).copy()
    M.has_canonical_format = True
    assert_equal(True, M.has_canonical_format)
    assert_equal(2, len(M.indices))
    M.sum_duplicates()
    assert_equal(2, len(M.indices))