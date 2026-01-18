from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing.glsl import GLSLPrinter
from sympy.printing.str import StrPrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.core import Tuple
from sympy.printing.glsl import glsl_code
import textwrap
def test_glsl_code_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    s = 'for (int i=0; i<m; i++){\n   y[i] = 0.0;\n}\nfor (int i=0; i<m; i++){\n   for (int j=0; j<n; j++){\n      for (int k=0; k<o; k++){\n         for (int l=0; l<p; l++){\n            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i * n * o * p + j * o * p + k * p + l, i * n * o * p + j * o * p + k * p + l, j * o * p + k * p + l) + '         }\n      }\n   }\n}'
    c = glsl_code((a[i, j, k, l] + b[i, j, k, l]) * c[j, k, l], assign_to=y[i])
    assert c == s