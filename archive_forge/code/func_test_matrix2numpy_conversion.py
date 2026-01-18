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
def test_matrix2numpy_conversion():
    a = Matrix([[1, 2, sin(x)], [x ** 2, x, Rational(1, 2)]])
    b = array([[1, 2, sin(x)], [x ** 2, x, Rational(1, 2)]])
    assert (matrix2numpy(a) == b).all()
    assert matrix2numpy(a).dtype == numpy.dtype('object')
    c = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='int8')
    d = matrix2numpy(Matrix([[1, 2], [10, 20]]), dtype='float64')
    assert c.dtype == numpy.dtype('int8')
    assert d.dtype == numpy.dtype('float64')