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
def test_elementwise_divide(self):
    expected = [[1, np.nan, np.nan, 1], [1, np.nan, 1, np.nan], [np.nan, 1, np.nan, np.nan]]
    assert_array_equal(toarray(self.datsp / self.datsp), expected)
    denom = self.spcreator(matrix([[1, 0, 0, 4], [-1, 0, 0, 0], [0, 8, 0, -5]], 'd'))
    expected = [[1, np.nan, np.nan, 0.5], [-3, np.nan, inf, np.nan], [np.nan, 0.25, np.nan, 0]]
    assert_array_equal(toarray(self.datsp / denom), expected)
    A = array([[1 - 2j, 0 + 5j, -1 + 0j], [4 - 3j, -3 + 6j, 5]])
    B = array([[5 + 2j, 7 - 3j, -2 + 1j], [0 - 1j, -4 + 2j, 9]])
    Asp = self.spcreator(A)
    Bsp = self.spcreator(B)
    assert_almost_equal(toarray(Asp / Bsp), A / B)
    A = array([[1, 2, 3], [-3, 2, 1]])
    B = array([[0, 1, 2], [0, -2, 3]])
    Asp = self.spcreator(A)
    Bsp = self.spcreator(B)
    with np.errstate(divide='ignore'):
        assert_array_equal(toarray(Asp / Bsp), A / B)
    A = array([[0, 1], [1, 0]])
    B = array([[1, 0], [1, 0]])
    Asp = self.spcreator(A)
    Bsp = self.spcreator(B)
    with np.errstate(divide='ignore', invalid='ignore'):
        assert_array_equal(np.array(toarray(Asp / Bsp)), A / B)