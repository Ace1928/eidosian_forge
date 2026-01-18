from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_boolean():
    assert jscode(x & y) == 'x && y'
    assert jscode(x | y) == 'x || y'
    assert jscode(~x) == '!x'
    assert jscode(x & y & z) == 'x && y && z'
    assert jscode(x | y | z) == 'x || y || z'
    assert jscode(x & y | z) == 'z || x && y'
    assert jscode((x | y) & z) == 'z && (x || y)'