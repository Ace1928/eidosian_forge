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
def test_glsl_code_constants_other():
    assert glsl_code(2 * GoldenRatio) == 'float GoldenRatio = 1.61803399;\n2*GoldenRatio'
    assert glsl_code(2 * Catalan) == 'float Catalan = 0.915965594;\n2*Catalan'
    assert glsl_code(2 * EulerGamma) == 'float EulerGamma = 0.577215665;\n2*EulerGamma'