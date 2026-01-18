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
def test_large_dimensions_reshape(self):
    mat1 = coo_matrix(([1], ([3000000], [1000])), (3000001, 1001))
    mat2 = coo_matrix(([1], ([1000], [3000000])), (1001, 3000001))
    assert_((mat1.reshape((1001, 3000001), order='C') != mat2).nnz == 0)
    assert_((mat2.reshape((3000001, 1001), order='F') != mat1).nnz == 0)