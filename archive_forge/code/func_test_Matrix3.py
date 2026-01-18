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
def test_Matrix3():
    a = array([[2, 4], [5, 1]])
    assert Matrix(a) == Matrix([[2, 4], [5, 1]])
    assert Matrix(a) != Matrix([[2, 4], [5, 2]])
    a = array([[sin(2), 4], [5, 1]])
    assert Matrix(a) == Matrix([[sin(2), 4], [5, 1]])
    assert Matrix(a) != Matrix([[sin(0), 4], [5, 1]])