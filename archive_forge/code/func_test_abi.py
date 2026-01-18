import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_abi(self):
    A = eye(2)
    b = ones(2)
    r_x, r_info = gmres(A, b)
    r_x = r_x.astype(complex)
    x, info = gmres(A.astype(complex), b.astype(complex))
    assert iscomplexobj(x)
    assert_allclose(r_x, x)
    assert r_info == info