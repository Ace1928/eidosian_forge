from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2 * x))
    assert jscode(g(x)) == '2*x'
    g = implemented_function('g', Lambda(x, 2 * x / Catalan))
    assert jscode(g(x)) == 'var Catalan = %s;\n2*x/Catalan' % Catalan.evalf(17)
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x * (1 + x) * (2 + x)))
    assert jscode(g(A[i]), assign_to=A[i]) == 'for (var i=0; i<n; i++){\n   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n}'