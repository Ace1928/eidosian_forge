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
def test_ellipsis_slicing(self):
    b = asmatrix(arange(50).reshape(5, 10))
    a = self.spcreator(b)
    assert_array_equal(a[...].toarray(), b[...].A)
    assert_array_equal(a[...,].toarray(), b[...,].A)
    assert_array_equal(a[1, ...].toarray(), b[1, ...].A)
    assert_array_equal(a[..., 1].toarray(), b[..., 1].A)
    assert_array_equal(a[1:, ...].toarray(), b[1:, ...].A)
    assert_array_equal(a[..., 1:].toarray(), b[..., 1:].A)
    assert_array_equal(a[1:, 1, ...].toarray(), b[1:, 1, ...].A)
    assert_array_equal(a[1, ..., 1:].toarray(), b[1, ..., 1:].A)
    assert_equal(a[1, 1, ...], b[1, 1, ...])
    assert_equal(a[1, ..., 1], b[1, ..., 1])