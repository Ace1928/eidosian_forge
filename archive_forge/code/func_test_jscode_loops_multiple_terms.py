from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    s0 = 'for (var i=0; i<m; i++){\n   y[i] = 0;\n}\n'
    s1 = 'for (var i=0; i<m; i++){\n   for (var j=0; j<n; j++){\n      for (var k=0; k<o; k++){\n         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i * n * o + j * o + k) + '      }\n   }\n}\n'
    s2 = 'for (var i=0; i<m; i++){\n   for (var k=0; k<o; k++){\n      y[i] = a[%s]*b[k] + y[i];\n' % (i * o + k) + '   }\n}\n'
    s3 = 'for (var i=0; i<m; i++){\n   for (var j=0; j<n; j++){\n      y[i] = a[%s]*b[j] + y[i];\n' % (i * n + j) + '   }\n}\n'
    c = jscode(b[j] * a[i, j] + b[k] * a[i, k] + b[j] * b[k] * c[i, j, k], assign_to=y[i])
    assert c == s0 + s1 + s2 + s3[:-1] or c == s0 + s1 + s3 + s2[:-1] or c == s0 + s2 + s1 + s3[:-1] or (c == s0 + s2 + s3 + s1[:-1]) or (c == s0 + s3 + s1 + s2[:-1]) or (c == s0 + s3 + s2 + s1[:-1])