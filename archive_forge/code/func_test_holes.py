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
def test_holes():
    nan = Undefined
    assert Piecewise((1, x < 2)).integrate(x) == Piecewise((x, x < 2), (nan, True))
    assert Piecewise((1, And(x > 1, x < 2))).integrate(x) == Piecewise((nan, x < 1), (x, x < 2), (nan, True))
    assert Piecewise((1, And(x > 1, x < 2))).integrate((x, 0, 3)) is nan
    assert Piecewise((1, And(x > 0, x < 4))).integrate((x, 1, 3)) == 2
    A, B = symbols('A B')
    a, b = symbols('a b', real=True)
    assert Piecewise((A, And(x < 0, a < 1)), (B, Or(x < 1, a > 2))).integrate(x) == Piecewise((B * x, a > 2), (Piecewise((A * x, x < 0), (B * x, x < 1), (nan, True)), a < 1), (Piecewise((B * x, x < 1), (nan, True)), True))