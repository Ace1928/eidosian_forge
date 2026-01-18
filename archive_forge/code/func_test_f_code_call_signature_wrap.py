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
def test_f_code_call_signature_wrap():
    x = symbols('x:20')
    expr = 0
    for sym in x:
        expr += sym
    routine = make_routine('test', expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = 'REAL*8 function test(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, &\n      x19, x2, x3, x4, x5, x6, x7, x8, x9)\nimplicit none\nREAL*8, intent(in) :: x0\nREAL*8, intent(in) :: x1\nREAL*8, intent(in) :: x10\nREAL*8, intent(in) :: x11\nREAL*8, intent(in) :: x12\nREAL*8, intent(in) :: x13\nREAL*8, intent(in) :: x14\nREAL*8, intent(in) :: x15\nREAL*8, intent(in) :: x16\nREAL*8, intent(in) :: x17\nREAL*8, intent(in) :: x18\nREAL*8, intent(in) :: x19\nREAL*8, intent(in) :: x2\nREAL*8, intent(in) :: x3\nREAL*8, intent(in) :: x4\nREAL*8, intent(in) :: x5\nREAL*8, intent(in) :: x6\nREAL*8, intent(in) :: x7\nREAL*8, intent(in) :: x8\nREAL*8, intent(in) :: x9\ntest = x0 + x1 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + &\n      x19 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9\nend function\n'
    assert source == expected