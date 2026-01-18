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
def test_c_code_argument_order():
    x, y, z = symbols('x,y,z')
    expr = x + y
    routine = make_routine('test', expr, argument_sequence=[z, x, y])
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = '#include "file.h"\n#include <math.h>\ndouble test(double z, double x, double y) {\n   double test_result;\n   test_result = x + y;\n   return test_result;\n}\n'
    assert source == expected