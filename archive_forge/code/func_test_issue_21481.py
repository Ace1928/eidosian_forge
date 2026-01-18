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
def test_issue_21481():
    b, e = symbols('b e')
    C = Piecewise((2, (b > 1) & (e > 0) | (b > 0) & (b < 1) & (e < 0) | (e >= 2) & (b < -1) & Eq(Mod(e, 2), 0) | (e <= -2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0)), (S.Half, (b > 1) & (e < 0) | (b > 0) & (e > 0) & (b < 1) | (e <= -2) & (b < -1) & Eq(Mod(e, 2), 0) | (e >= 2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0)), (-S.Half, Eq(Mod(e, 2), 1) & ((e <= -1) & (b < -1) | (e >= 1) & (b > -1) & (b < 0))), (-2, (e >= 1) & (b < -1) & Eq(Mod(e, 2), 1) | (e <= -1) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 1)))
    A = Piecewise((1, Eq(b, 1) | Eq(e, 0) | Eq(b, -1) & Eq(Mod(e, 2), 0)), (0, Eq(b, 0) & (e > 0)), (-1, Eq(b, -1) & Eq(Mod(e, 2), 1)), (C, Eq(im(b), 0) & Eq(im(e), 0)))
    B = piecewise_fold(A)
    sa = A.simplify()
    sb = B.simplify()
    v = (-2, -1, -S.Half, 0, S.Half, 1, 2)
    for i in v:
        for j in v:
            r = {b: i, e: j}
            ok = [k.xreplace(r) for k in (A, B, sa, sb)]
            assert len(set(ok)) == 1