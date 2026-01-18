from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_julia_matrix_assign_to_more():
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert julia_code(A, assign_to=B) == 'B = [1 2 3]'
    raises(ValueError, lambda: julia_code(A, assign_to=x))
    raises(ValueError, lambda: julia_code(A, assign_to=C))