from io import StringIO
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
def test_empty_c_header():
    code_gen = C99CodeGen()
    source = get_string(code_gen.dump_h, [])
    assert source == '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n#endif\n'