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
def test_1xN_vecs():
    gl = glsl_code
    for i in range(1, 10):
        A = Matrix(range(i))
        assert gl(A.transpose()) == gl(A)
        assert gl(A, mat_transpose=True) == gl(A)
        if i > 1:
            if i <= 4:
                assert gl(A) == 'vec%s(%s)' % (i, ', '.join((str(s) for s in range(i))))
            else:
                assert gl(A) == 'float[%s](%s)' % (i, ', '.join((str(s) for s in range(i))))