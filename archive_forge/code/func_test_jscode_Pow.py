from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_Pow():
    g = implemented_function('g', Lambda(x, 2 * x))
    assert jscode(x ** 3) == 'Math.pow(x, 3)'
    assert jscode(x ** y ** 3) == 'Math.pow(x, Math.pow(y, 3))'
    assert jscode(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == 'Math.pow(3.5*2*x, -x + Math.pow(y, x))/(Math.pow(x, 2) + y)'
    assert jscode(x ** (-1.0)) == '1/x'