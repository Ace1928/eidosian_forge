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
def test_fancy_assign_list(self):
    np.random.seed(1234)
    D = asmatrix(np.random.rand(5, 7))
    S = self.spcreator(D)
    X = np.random.rand(2, 3)
    I = [[1, 2, 3], [3, 4, 2]]
    J = [[5, 6, 3], [2, 3, 1]]
    S[I, J] = X
    D[I, J] = X
    assert_equal(S.toarray(), D)
    I_bad = [[ii + 5 for ii in i] for i in I]
    J_bad = [[jj + 7 for jj in j] for j in J]
    C = [1, 2, 3]
    S[I, J] = C
    D[I, J] = C
    assert_equal(S.toarray(), D)
    S[I, J] = 3
    D[I, J] = 3
    assert_equal(S.toarray(), D)
    assert_raises(IndexError, S.__setitem__, (I_bad, J), C)
    assert_raises(IndexError, S.__setitem__, (I, J_bad), C)