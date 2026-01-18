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
def test_Matrices_1x7_spread_assign_to_symbols():
    gl = glsl_code
    A = Matrix([1, 2, 3, 4, 5, 6, 7])
    assign_to = symbols('x.a x.b x.c x.d x.e x.f x.g')
    assert gl(A, assign_to=assign_to) == textwrap.dedent('        x.a = 1;\n        x.b = 2;\n        x.c = 3;\n        x.d = 4;\n        x.e = 5;\n        x.f = 6;\n        x.g = 7;')