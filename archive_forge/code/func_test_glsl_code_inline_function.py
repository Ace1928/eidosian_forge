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
def test_glsl_code_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2 * x))
    assert glsl_code(g(x)) == '2*x'
    g = implemented_function('g', Lambda(x, 2 * x / Catalan))
    assert glsl_code(g(x)) == 'float Catalan = 0.915965594;\n2*x/Catalan'
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x * (1 + x) * (2 + x)))
    assert glsl_code(g(A[i]), assign_to=A[i]) == 'for (int i=0; i<n; i++){\n   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n}'