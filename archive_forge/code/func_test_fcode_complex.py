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
def test_fcode_complex():
    import sympy.utilities.codegen
    sympy.utilities.codegen.COMPLEX_ALLOWED = True
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    result = codegen(('test', x + y), 'f95', 'test', header=False, empty=False)
    source = result[0][1]
    expected = 'REAL*8 function test(x, y)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\ntest = x + y\nend function\n'
    assert source == expected
    x = Symbol('x')
    y = Symbol('y', real=True)
    result = codegen(('test', x + y), 'f95', 'test', header=False, empty=False)
    source = result[0][1]
    expected = 'COMPLEX*16 function test(x, y)\nimplicit none\nCOMPLEX*16, intent(in) :: x\nREAL*8, intent(in) :: y\ntest = x + y\nend function\n'
    assert source == expected
    sympy.utilities.codegen.COMPLEX_ALLOWED = False