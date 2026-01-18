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
def test_issue_14240():
    assert piecewise_fold(Piecewise((1, a), (2, b), (4, True)) + Piecewise((8, a), (16, True))) == Piecewise((9, a), (18, b), (20, True))
    assert piecewise_fold(Piecewise((2, a), (3, b), (5, True)) * Piecewise((7, a), (11, True))) == Piecewise((14, a), (33, b), (55, True))
    assert piecewise_fold(Add(*[Piecewise((i, a), (0, True)) for i in range(40)])) == Piecewise((780, a), (0, True))
    assert piecewise_fold(Mul(*[Piecewise((i, a), (0, True)) for i in range(1, 41)])) == Piecewise((factorial(40), a), (0, True))