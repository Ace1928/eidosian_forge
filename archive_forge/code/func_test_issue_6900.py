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
def test_issue_6900():
    from itertools import permutations
    t0, t1, T, t = symbols('t0, t1 T t')
    f = Piecewise((0, t < t0), (x, And(t0 <= t, t < t1)), (0, t >= t1))
    g = f.integrate(t)
    assert g == Piecewise((0, t <= t0), (t * x - t0 * x, t <= Max(t0, t1)), (-t0 * x + x * Max(t0, t1), True))
    for i in permutations(range(2)):
        reps = dict(zip((t0, t1), i))
        for tt in range(-1, 3):
            assert g.xreplace(reps).subs(t, tt) == f.xreplace(reps).integrate(t).subs(t, tt)
    lim = Tuple(t, t0, T)
    g = f.integrate(lim)
    ans = Piecewise((-t0 * x + x * Min(T, Max(t0, t1)), T > t0), (0, True))
    for i in permutations(range(3)):
        reps = dict(zip((t0, t1, T), i))
        tru = f.xreplace(reps).integrate(lim.xreplace(reps))
        assert tru == ans.xreplace(reps)
    assert g == ans