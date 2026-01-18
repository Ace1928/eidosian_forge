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
def test_issue_10258():
    assert Piecewise((0, x < 1), (1, True)).is_zero is None
    assert Piecewise((-1, x < 1), (1, True)).is_zero is False
    a = Symbol('a', zero=True)
    assert Piecewise((0, x < 1), (a, True)).is_zero
    assert Piecewise((1, x < 1), (a, x < 3)).is_zero is None
    a = Symbol('a')
    assert Piecewise((0, x < 1), (a, True)).is_zero is None
    assert Piecewise((0, x < 1), (1, True)).is_nonzero is None
    assert Piecewise((1, x < 1), (2, True)).is_nonzero
    assert Piecewise((0, x < 1), (oo, True)).is_finite is None
    assert Piecewise((0, x < 1), (1, True)).is_finite
    b = Basic()
    assert Piecewise((b, x < 1)).is_finite is None
    c = Piecewise((1, x < 0), (2, True)) < 3
    assert c != True
    assert piecewise_fold(c) == True