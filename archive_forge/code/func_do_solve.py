from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from platform import python_implementation
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres
def do_solve(**kw):
    count[0] = 0
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, '.*called without specifying.*')
        x0, flag = lgmres(A, b, x0=zeros(A.shape[0]), inner_m=6, rtol=1e-14, **kw)
    count_0 = count[0]
    assert_(allclose(A @ x0, b, rtol=1e-12, atol=1e-12), norm(A @ x0 - b))
    return (x0, count_0)