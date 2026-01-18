import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
def test_isolve_gmres(matrices):
    A_dense, A_sparse, b = matrices
    x, info = splin.gmres(A_sparse, b, atol=1e-15)
    assert info == 0
    assert isinstance(x, np.ndarray)
    assert_allclose(A_sparse @ x, b)