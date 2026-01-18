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
def test_constructor_smallcol(self):
    data = arange(6) + 1
    col = array([1, 2, 1, 0, 0, 2], dtype=np.int64)
    ptr = array([0, 2, 4, 6], dtype=np.int64)
    a = csr_matrix((data, col, ptr), shape=(3, 3))
    b = array([[0, 1, 2], [4, 3, 0], [5, 0, 6]], 'd')
    assert_equal(a.indptr.dtype, np.dtype(np.int32))
    assert_equal(a.indices.dtype, np.dtype(np.int32))
    assert_array_equal(a.toarray(), b)