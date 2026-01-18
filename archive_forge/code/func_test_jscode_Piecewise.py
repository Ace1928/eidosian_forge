from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_Piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))
    p = jscode(expr)
    s = '((x < 1) ? (\n   x\n)\n: (\n   Math.pow(x, 2)\n))'
    assert p == s
    assert jscode(expr, assign_to='c') == 'if (x < 1) {\n   c = x;\n}\nelse {\n   c = Math.pow(x, 2);\n}'
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: jscode(expr))