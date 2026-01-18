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
def test_matrix2numpy():
    a = matrix2numpy(Matrix([[1, x ** 2], [3 * sin(x), 0]]))
    assert isinstance(a, ndarray)
    assert a.shape == (2, 2)
    assert a[0, 0] == 1
    assert a[0, 1] == x ** 2
    assert a[1, 0] == 3 * sin(x)
    assert a[1, 1] == 0