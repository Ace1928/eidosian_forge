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
def test_issue_22917():
    p = Piecewise((0, ITE((x - y > 1) | (2 * x - 2 * y > 1), False, ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))), (Piecewise((0, ITE(x - y > 1, True, 2 * x - 2 * y > 1)), (2 * Piecewise((0, x - y > 1), (y, True)), True)), True)) + 2 * Piecewise((1, ITE((x - y > 1) | (2 * x - 2 * y > 1), False, ITE(x - y > 1, 2 * y - 2 < -1, 2 * x - 2 * y > 1))), (Piecewise((1, ITE(x - y > 1, True, 2 * x - 2 * y > 1)), (2 * Piecewise((1, x - y > 1), (x, True)), True)), True))
    assert piecewise_fold(p) == Piecewise((2, (x - y > S.Half) | (x - y > 1)), (2 * y + 4, x - y > 1), (4 * x + 2 * y, True))
    assert piecewise_fold(p > 1).rewrite(ITE) == ITE((x - y > S.Half) | (x - y > 1), True, ITE(x - y > 1, 2 * y + 4 > 1, 4 * x + 2 * y > 1))