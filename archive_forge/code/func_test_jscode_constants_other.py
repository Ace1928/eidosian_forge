from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.testing.pytest import raises
from sympy.printing.jscode import JavascriptCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.jscode import jscode
def test_jscode_constants_other():
    assert jscode(2 * GoldenRatio) == 'var GoldenRatio = %s;\n2*GoldenRatio' % GoldenRatio.evalf(17)
    assert jscode(2 * Catalan) == 'var Catalan = %s;\n2*Catalan' % Catalan.evalf(17)
    assert jscode(2 * EulerGamma) == 'var EulerGamma = %s;\n2*EulerGamma' % EulerGamma.evalf(17)