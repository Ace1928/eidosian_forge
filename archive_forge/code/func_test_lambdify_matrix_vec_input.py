from sympy.external.importtools import version_tuple
from sympy.external import import_module
from sympy.core.numbers import (Float, Integer, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import (Matrix, list2numpy, matrix2numpy, symarray)
from sympy.utilities.lambdify import lambdify
import sympy
import mpmath
from sympy.abc import x, y, z
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.testing.pytest import raises
def test_lambdify_matrix_vec_input():
    X = sympy.DeferredVector('X')
    M = Matrix([[X[0] ** 2, X[0] * X[1], X[0] * X[2]], [X[1] * X[0], X[1] ** 2, X[1] * X[2]], [X[2] * X[0], X[2] * X[1], X[2] ** 2]])
    f = lambdify(X, M, [{'ImmutableMatrix': numpy.array}, 'numpy'])
    Xh = array([1.0, 2.0, 3.0])
    expected = array([[Xh[0] ** 2, Xh[0] * Xh[1], Xh[0] * Xh[2]], [Xh[1] * Xh[0], Xh[1] ** 2, Xh[1] * Xh[2]], [Xh[2] * Xh[0], Xh[2] * Xh[1], Xh[2] ** 2]])
    actual = f(Xh)
    assert numpy.allclose(actual, expected)