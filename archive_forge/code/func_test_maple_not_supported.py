from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import besseli
from sympy.printing.maple import maple_code
def test_maple_not_supported():
    assert maple_code(S.ComplexInfinity) == '# Not supported in maple:\n# ComplexInfinity\nzoo'