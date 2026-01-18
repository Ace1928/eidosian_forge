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
@with_64bit_maxval_limit(fixed_dtype=np.int64)
def check_unlimited():
    a = csc_matrix([[1, 2], [3, 4], [5, 6]])
    a.getnnz(axis=1)
    a.sum(axis=0)
    a = csr_matrix([[1, 2, 3], [3, 4, 6]])
    a.getnnz(axis=0)
    a = coo_matrix([[1, 2, 3], [3, 4, 5]])
    a.getnnz(axis=0)