from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import besseli
from sympy.printing.maple import maple_code
def test_maple_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x ** 2, True))
    assert maple_code(2 * pw) == '2*piecewise(x < 1, x, x^2)'
    assert maple_code(pw / x) == 'piecewise(x < 1, x, x^2)/x'
    assert maple_code(pw / (x * y)) == 'piecewise(x < 1, x, x^2)/(x*y)'
    assert maple_code(pw / 3) == '(1/3)*piecewise(x < 1, x, x^2)'