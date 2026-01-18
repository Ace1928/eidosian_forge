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
def test_maximum_minimum(self):
    A_dense = np.array([[1, 0, 3], [0, 4, 5], [0, 0, 0]])
    B_dense = np.array([[1, 1, 2], [0, 3, 6], [1, -1, 0]])
    A_dense_cpx = np.array([[1, 0, 3], [0, 4 + 2j, 5], [0, 1j, -1j]])

    def check(dtype, dtype2, btype):
        if np.issubdtype(dtype, np.complexfloating):
            A = self.spcreator(A_dense_cpx.astype(dtype))
        else:
            A = self.spcreator(A_dense.astype(dtype))
        if btype == 'scalar':
            B = dtype2.type(1)
        elif btype == 'scalar2':
            B = dtype2.type(-1)
        elif btype == 'dense':
            B = B_dense.astype(dtype2)
        elif btype == 'sparse':
            B = self.spcreator(B_dense.astype(dtype2))
        else:
            raise ValueError()
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Taking maximum .minimum. with > 0 .< 0. number results to a dense matrix')
            max_s = A.maximum(B)
            min_s = A.minimum(B)
        max_d = np.maximum(toarray(A), toarray(B))
        assert_array_equal(toarray(max_s), max_d)
        assert_equal(max_s.dtype, max_d.dtype)
        min_d = np.minimum(toarray(A), toarray(B))
        assert_array_equal(toarray(min_s), min_d)
        assert_equal(min_s.dtype, min_d.dtype)
    for dtype in self.math_dtypes:
        for dtype2 in [np.int8, np.float64, np.complex128]:
            for btype in ['scalar', 'scalar2', 'dense', 'sparse']:
                check(np.dtype(dtype), np.dtype(dtype2), btype)