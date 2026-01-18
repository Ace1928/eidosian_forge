import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_linearoperator_deallocation():
    M_d = np.eye(10)
    M_s = csc_matrix(M_d)
    M_o = aslinearoperator(M_d)
    with assert_deallocated(lambda: arpack.SpLuInv(M_s)):
        pass
    with assert_deallocated(lambda: arpack.LuInv(M_d)):
        pass
    with assert_deallocated(lambda: arpack.IterInv(M_s)):
        pass
    with assert_deallocated(lambda: arpack.IterOpInv(M_o, None, 0.3)):
        pass
    with assert_deallocated(lambda: arpack.IterOpInv(M_o, M_o, 0.3)):
        pass