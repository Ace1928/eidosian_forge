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
def test_issue_10137():
    a = Symbol('a', real=True)
    b = Symbol('b', real=True)
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    p0 = Piecewise((0, Or(x < a, x > b)), (1, True))
    p1 = Piecewise((0, Or(a > x, b < x)), (1, True))
    assert integrate(p0, (x, y, oo)) == integrate(p1, (x, y, oo))
    p3 = Piecewise((1, And(0 < x, x < a)), (0, True))
    p4 = Piecewise((1, And(a > x, x > 0)), (0, True))
    ip3 = integrate(p3, x)
    assert ip3 == Piecewise((0, x <= 0), (x, x <= Max(0, a)), (Max(0, a), True))
    ip4 = integrate(p4, x)
    assert ip4 == ip3
    assert p3.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2
    assert p4.integrate((x, 2, 4)) == Min(4, Max(2, a)) - 2