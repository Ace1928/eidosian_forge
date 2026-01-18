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
def test_Matrix_mul():
    M = Matrix([[1, 2, 3], [x, y, x]])
    with ignore_warnings(PendingDeprecationWarning):
        m = matrix([[2, 4], [x, 6], [x, z ** 2]])
    assert M * m == Matrix([[2 + 5 * x, 16 + 3 * z ** 2], [2 * x + x * y + x ** 2, 4 * x + 6 * y + x * z ** 2]])
    assert m * M == Matrix([[2 + 4 * x, 4 + 4 * y, 6 + 4 * x], [7 * x, 2 * x + 6 * y, 9 * x], [x + x * z ** 2, 2 * x + y * z ** 2, 3 * x + x * z ** 2]])
    a = array([2])
    assert a[0] * M == 2 * M
    assert M * a[0] == 2 * M