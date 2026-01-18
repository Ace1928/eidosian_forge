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
def test_mix_number_pow_symbols():
    assert mcode(pi ** 3) == 'pi^3'
    assert mcode(x ** 2) == 'x.^2'
    assert mcode(x ** pi ** 3) == 'x.^(pi^3)'
    assert mcode(x ** y) == 'x.^y'
    assert mcode(x ** y ** z) == 'x.^(y.^z)'
    assert mcode((x ** y) ** z) == '(x.^y).^z'