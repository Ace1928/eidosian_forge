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
def spcreator(self, D, sorted_indices=False, **kwargs):
    """Replace D with a non-canonical equivalent: containing
        duplicate elements and explicit zeros"""
    construct = super().spcreator
    M = construct(D, **kwargs)
    zero_pos = (M.toarray() == 0).nonzero()
    has_zeros = zero_pos[0].size > 0
    if has_zeros:
        k = zero_pos[0].size // 2
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            M = self._insert_explicit_zero(M, zero_pos[0][k], zero_pos[1][k])
    arg1 = self._arg1_for_noncanonical(M, sorted_indices)
    if 'shape' not in kwargs:
        kwargs['shape'] = M.shape
    NC = construct(arg1, **kwargs)
    if NC.dtype in [np.float32, np.complex64]:
        rtol = 1e-05
    else:
        rtol = 1e-07
    assert_allclose(NC.toarray(), M.toarray(), rtol=rtol)
    if has_zeros:
        assert_((NC.data == 0).any())
    return NC