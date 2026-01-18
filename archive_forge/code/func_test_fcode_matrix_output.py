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
def test_fcode_matrix_output():
    x, y, z = symbols('x,y,z')
    e1 = x + y
    e2 = Matrix([[x, y], [z, 16]])
    name_expr = ('test', (e1, e2))
    result = codegen(name_expr, 'f95', 'test', header=False, empty=False)
    source = result[0][1]
    expected = 'REAL*8 function test(x, y, z, out_%(hash)s)\nimplicit none\nREAL*8, intent(in) :: x\nREAL*8, intent(in) :: y\nREAL*8, intent(in) :: z\nREAL*8, intent(out), dimension(1:2, 1:2) :: out_%(hash)s\nout_%(hash)s(1, 1) = x\nout_%(hash)s(2, 1) = z\nout_%(hash)s(1, 2) = y\nout_%(hash)s(2, 2) = 16\ntest = x + y\nend function\n'
    a = source.splitlines()[5]
    b = a.split('_')
    out = b[1]
    expected = expected % {'hash': out}
    assert source == expected