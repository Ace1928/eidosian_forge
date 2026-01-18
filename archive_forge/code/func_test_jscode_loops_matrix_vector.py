from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    s = 'for (var i=0; i<m; i++){\n   y[i] = 0;\n}\nfor (var i=0; i<m; i++){\n   for (var j=0; j<n; j++){\n      y[i] = A[n*i + j]*x[j] + y[i];\n   }\n}'
    c = jscode(A[i, j] * x[j], assign_to=y[i])
    assert c == s