from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import besseli
from sympy.printing.maple import maple_code
def test_automatic_rewrites():
    assert maple_code(lucas(x)) == '2^(-x)*((1 - sqrt(5))^x + (1 + sqrt(5))^x)'
    assert maple_code(sinc(x)) == 'piecewise(x <> 0, sin(x)/x, 1)'