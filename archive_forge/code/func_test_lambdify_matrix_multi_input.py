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
def test_lambdify_matrix_multi_input():
    M = sympy.Matrix([[x ** 2, x * y, x * z], [y * x, y ** 2, y * z], [z * x, z * y, z ** 2]])
    f = lambdify((x, y, z), M, [{'ImmutableMatrix': numpy.array}, 'numpy'])
    xh, yh, zh = (1.0, 2.0, 3.0)
    expected = array([[xh ** 2, xh * yh, xh * zh], [yh * xh, yh ** 2, yh * zh], [zh * xh, zh * yh, zh ** 2]])
    actual = f(xh, yh, zh)
    assert numpy.allclose(actual, expected)