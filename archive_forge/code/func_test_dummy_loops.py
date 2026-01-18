from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.sets import Range
from sympy.logic import ITE
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises
from sympy.printing.rcode import RCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.rcode import rcode
def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = 'for (i_%(icount)i in 1:m_%(mcount)i){\n   y[i_%(icount)i] = x[i_%(icount)i];\n}' % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = rcode(x[i], assign_to=y[i])
    assert code == expected