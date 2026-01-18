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
def test_inplace_ops(self):
    A = lil_matrix([[0, 2, 3], [4, 0, 6]])
    B = lil_matrix([[0, 1, 0], [0, 2, 3]])
    data = {'add': (B, A + B), 'sub': (B, A - B), 'mul': (3, A * 3)}
    for op, (other, expected) in data.items():
        result = A.copy()
        getattr(result, '__i%s__' % op)(other)
        assert_array_equal(result.toarray(), expected.toarray())
    A = lil_matrix((1, 3), dtype=np.dtype('float64'))
    B = array([0.1, 0.1, 0.1])
    A[0, :] += B
    assert_array_equal(A[0, :].toarray().squeeze(), B)