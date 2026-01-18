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
def test_fancy_indexing_multidim_set(self):
    n, m = (5, 10)

    def _test_set_slice(i, j):
        A = self.spcreator((n, m))
        with check_remains_sorted(A), suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            A[i, j] = 1
        B = asmatrix(np.zeros((n, m)))
        B[i, j] = 1
        assert_array_almost_equal(A.toarray(), B)
    for i, j in [(np.array([[1, 2], [1, 3]]), [1, 3]), (np.array([0, 4]), [[0, 3], [1, 2]]), ([[1, 2, 3], [0, 2, 4]], [[0, 4, 3], [4, 1, 2]])]:
        _test_set_slice(i, j)