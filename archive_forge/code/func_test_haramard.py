from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.testing.pytest import XFAIL
from sympy.printing.julia import julia_code
def test_haramard():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    assert julia_code(C) == 'A .* B'
    assert julia_code(C * v) == '(A .* B) * v'
    assert julia_code(h * C * v) == 'h * (A .* B) * v'
    assert julia_code(C * A) == '(A .* B) * A'
    assert julia_code(C * x * y) == '(x .* y) * (A .* B)'