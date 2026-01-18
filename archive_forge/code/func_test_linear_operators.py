from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._lsq.common import (
def test_linear_operators():
    A = np.arange(6).reshape((3, 2))
    d_left = np.array([-1, 2, 5])
    DA = np.diag(d_left).dot(A)
    J_left = left_multiplied_operator(A, d_left)
    d_right = np.array([5, 10])
    AD = A.dot(np.diag(d_right))
    J_right = right_multiplied_operator(A, d_right)
    x = np.array([-2, 3])
    X = -2 * np.arange(2, 8).reshape((2, 3))
    xt = np.array([0, -2, 15])
    assert_allclose(DA.dot(x), J_left.matvec(x))
    assert_allclose(DA.dot(X), J_left.matmat(X))
    assert_allclose(DA.T.dot(xt), J_left.rmatvec(xt))
    assert_allclose(AD.dot(x), J_right.matvec(x))
    assert_allclose(AD.dot(X), J_right.matmat(X))
    assert_allclose(AD.T.dot(xt), J_right.rmatvec(xt))