from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import (arg, atan2, bernoulli, beta, ceiling, chebyshevu,
from sympy.functions import (sin, cos, tan, cot, sec, csc, asin, acos, acot,
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
from sympy.functions.special.error_functions import (Chi, Ci, erf, erfc, erfi,
from sympy.printing.octave import octave_code, octave_code as mcode
def test_KroneckerDelta():
    from sympy.functions import KroneckerDelta
    assert mcode(KroneckerDelta(x, y)) == 'double(x == y)'
    assert mcode(KroneckerDelta(x, y + 1)) == 'double(x == (y + 1))'
    assert mcode(KroneckerDelta(2 ** x, y)) == 'double((2.^x) == y)'