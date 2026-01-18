from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.function import (Function, diff, expand)
from sympy.core.mul import Mul
from sympy.core.mod import Mod
from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, transpose)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import (Piecewise,
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import (Integral, integrate)
from sympy.logic.boolalg import (And, ITE, Not, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.printing import srepr
from sympy.sets.contains import Contains
from sympy.sets.sets import Interval
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.utilities.lambdify import lambdify
def test_Piecewise_rewrite_as_ITE():
    a, b, c, d = symbols('a:d')

    def _ITE(*args):
        return Piecewise(*args).rewrite(ITE)
    assert _ITE((a, x < 1), (b, x >= 1)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < oo)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, Or(y < 1, x < oo)), (c, y > 0)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, True)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (b, x < 2), (c, True)) == ITE(x < 1, a, ITE(x < 2, b, c))
    assert _ITE((a, x < 1), (b, y < 2), (c, True)) == ITE(x < 1, a, ITE(y < 2, b, c))
    assert _ITE((a, x < 1), (b, x < oo), (c, y < 1)) == ITE(x < 1, a, b)
    assert _ITE((a, x < 1), (c, y < 1), (b, x < oo), (d, True)) == ITE(x < 1, a, ITE(y < 1, c, b))
    assert _ITE((a, x < 0), (b, Or(x < oo, y < 1))) == ITE(x < 0, a, b)
    raises(TypeError, lambda: _ITE((x + 1, x < 1), (x, True)))
    raises(NotImplementedError, lambda: _ITE((x, x < y), (y, x >= a)))
    raises(ValueError, lambda: _ITE((a, x < 2), (b, x > 3)))