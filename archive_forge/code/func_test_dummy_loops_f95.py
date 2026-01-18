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
def test_dummy_loops_f95():
    from sympy.tensor import IndexedBase, Idx
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = 'subroutine test_dummies(m_%(mcount)i, x, y)\nimplicit none\nINTEGER*4, intent(in) :: m_%(mcount)i\nREAL*8, intent(in), dimension(1:m_%(mcount)i) :: x\nREAL*8, intent(out), dimension(1:m_%(mcount)i) :: y\nINTEGER*4 :: i_%(icount)i\ndo i_%(icount)i = 1, m_%(mcount)i\n   y(i_%(icount)i) = x(i_%(icount)i)\nend do\nend subroutine\n' % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    r = make_routine('test_dummies', Eq(y[i], x[i]))
    c = FCodeGen()
    code = get_string(c.dump_f95, [r])
    assert code == expected