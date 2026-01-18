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
def test_issue_5227():
    f = 0.0032513612725229 * Piecewise((0, x < -80.8461538461539), (-0.0160799238820171 * x + 1.33215984776403, x < 2), (Piecewise((0.3, x > 123), (0.7, True)) + Piecewise((0.4, x > 2), (0.6, True)), x <= 123), (-0.00817409766454352 * x + 2.10541401273885, x < 380.571428571429), (0, True))
    i = integrate(f, (x, -oo, oo))
    assert i == Integral(f, (x, -oo, oo)).doit()
    assert str(i) == '1.00195081676351'
    assert Piecewise((1, x - y < 0), (0, True)).integrate(y) == Piecewise((0, y <= x), (-x + y, True))