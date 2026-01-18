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
def test_fancy_indexing_set(self):
    n, m = (5, 10)

    def _test_set_slice(i, j):
        A = self.spcreator((n, m))
        B = asmatrix(np.zeros((n, m)))
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            B[i, j] = 1
            with check_remains_sorted(A):
                A[i, j] = 1
        assert_array_almost_equal(A.toarray(), B)
    for i, j in [((2, 3, 4), slice(None, 10, 4)), (np.arange(3), slice(5, -2)), (slice(2, 5), slice(5, -2))]:
        _test_set_slice(i, j)
    for i, j in [(np.arange(3), np.arange(3)), ((0, 3, 4), (1, 2, 4))]:
        _test_set_slice(i, j)