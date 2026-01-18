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
def test_glsl_code_Rational():
    assert glsl_code(Rational(3, 7)) == '3.0/7.0'
    assert glsl_code(Rational(18, 9)) == '2'
    assert glsl_code(Rational(3, -7)) == '-3.0/7.0'
    assert glsl_code(Rational(-3, -7)) == '3.0/7.0'