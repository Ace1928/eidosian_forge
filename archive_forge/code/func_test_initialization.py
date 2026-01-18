import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr
def test_initialization():
    b_copy = b.copy()
    x_ref = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit)
    x0 = np.zeros(x_ref[0].shape)
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    assert_array_equal(b_copy, b)
    assert_allclose(x_ref[0], x[0])
    x0 = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=1)[0]
    x = lsqr(G, b, show=show, atol=tol, btol=tol, iter_lim=maxit, x0=x0)
    assert_allclose(x_ref[0], x[0])
    assert_array_equal(b_copy, b)