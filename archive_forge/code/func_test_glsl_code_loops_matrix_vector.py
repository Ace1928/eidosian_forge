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
def test_glsl_code_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    s = 'for (int i=0; i<m; i++){\n   y[i] = 0.0;\n}\nfor (int i=0; i<m; i++){\n   for (int j=0; j<n; j++){\n      y[i] = A[n*i + j]*x[j] + y[i];\n   }\n}'
    c = glsl_code(A[i, j] * x[j], assign_to=y[i])
    assert c == s