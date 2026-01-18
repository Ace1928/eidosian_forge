from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import besseli
from sympy.printing.maple import maple_code
def test_maple_piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))
    assert maple_code(expr) == 'piecewise(x < 1, x, x^2)'
    assert maple_code(expr, assign_to='r') == 'r := piecewise(x < 1, x, x^2)'
    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = 'piecewise(x < 1, x^2, x < 2, x^3, x < 3, x^4, x^5)'
    assert maple_code(expr) == expected
    assert maple_code(expr, assign_to='r') == 'r := ' + expected
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: maple_code(expr))