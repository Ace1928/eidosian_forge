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
def test_multiple_results_c():
    x, y, z = symbols('x,y,z')
    expr1 = (x + y) * z
    expr2 = (x - y) * z
    routine = make_routine('test', [expr1, expr2])
    code_gen = C99CodeGen()
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))