import warnings
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_has_quadratic(self) -> None:
    """Test the has_quadratic_term function."""
    x = Variable()
    assert not x.has_quadratic_term()
    assert not (3 + 3 * x).has_quadratic_term()
    assert (x ** 2).has_quadratic_term()
    assert (x ** 2 / 2).has_quadratic_term()
    assert (x ** 2 + x ** 3).has_quadratic_term()
    assert (2 * x ** 2 + x ** 3).has_quadratic_term()
    assert cp.conj(x ** 2).has_quadratic_term()
    assert not cp.pos(x ** 2).has_quadratic_term()
    assert cp.square(x ** 2).has_quadratic_term()
    assert cp.huber(x ** 3).has_quadratic_term()
    assert cp.power(x ** 2, 1).has_quadratic_term()
    assert cp.quad_over_lin(x ** 3, 1).has_quadratic_term()
    assert not cp.quad_over_lin(x ** 3, x).has_quadratic_term()
    y = cp.Variable(2)
    P = np.eye(2)
    assert cp.matrix_frac(y ** 3, P).has_quadratic_term()
    P = cp.Parameter((2, 2), PSD=True)
    assert cp.matrix_frac(y ** 3, P).has_quadratic_term()