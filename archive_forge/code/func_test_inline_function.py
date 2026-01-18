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
def test_inline_function():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A, x, y = map(IndexedBase, 'Axy')
    i = Idx('i', m)
    p = FCodeGen()
    func = implemented_function('func', Lambda(n, n * (n + 1)))
    routine = make_routine('test_inline', Eq(y[i], func(x[i])))
    code = get_string(p.dump_f95, [routine])
    expected = 'subroutine test_inline(m, x, y)\nimplicit none\nINTEGER*4, intent(in) :: m\nREAL*8, intent(in), dimension(1:m) :: x\nREAL*8, intent(out), dimension(1:m) :: y\nINTEGER*4 :: i\ndo i = 1, m\n   y(i) = %s*%s\nend do\nend subroutine\n'
    args = ('x(i)', '(x(i) + 1)')
    assert code == expected % args or code == expected % args[::-1]