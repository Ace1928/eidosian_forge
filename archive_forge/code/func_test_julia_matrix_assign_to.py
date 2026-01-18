from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_julia_matrix_assign_to():
    A = Matrix([[1, 2, 3]])
    assert julia_code(A, assign_to='a') == 'a = [1 2 3]'
    A = Matrix([[1, 2], [3, 4]])
    assert julia_code(A, assign_to='A') == 'A = [1 2;\n3 4]'