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
def test_fancy_assignment_dtypes(self):

    def check(dtype):
        A = self.spcreator((5, 5), dtype=dtype)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            A[[0, 1], [0, 1]] = dtype.type(1)
            assert_equal(A.sum(), dtype.type(1) * 2)
            A[0:2, 0:2] = dtype.type(1.0)
            assert_equal(A.sum(), dtype.type(1) * 4)
            A[2, 2] = dtype.type(1.0)
            assert_equal(A.sum(), dtype.type(1) * 4 + dtype.type(1))
    for dtype in supported_dtypes:
        check(np.dtype(dtype))