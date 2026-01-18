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
def test_fancy_indexing_2d_assign(self):
    mat = self.spcreator(array([[1, 0], [2, 3]]))
    with suppress_warnings() as sup:
        sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure')
        mat[[0, 1], [1, 1]] = mat[[1, 0], [0, 0]]
    assert_equal(toarray(mat), array([[1, 2], [2, 1]]))