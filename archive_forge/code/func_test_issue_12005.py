from sympy.concrete.summations import Sum
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import clear_cache
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr, unchanged
from sympy.core.function import (Subs, Function, diff, Lambda, expand,
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, acos
from sympy.functions.special.error_functions import expint
from sympy.functions.special.gamma_functions import loggamma, polygamma
from sympy.matrices.dense import Matrix
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.tensor.indexed import Indexed
from sympy.core.function import (PoleError, _mexpand, arity,
from sympy.core.parameters import _exp_is_pow
from sympy.core.sympify import sympify, SympifyError
from sympy.matrices import MutableMatrix, ImmutableMatrix
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import subsets, variations
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow
from sympy.abc import t, w, x, y, z
def test_issue_12005():
    e1 = Subs(Derivative(f(x), x), x, x)
    assert e1.diff(x) == Derivative(f(x), x, x)
    e2 = Subs(Derivative(f(x), x), x, x ** 2 + 1)
    assert e2.diff(x) == 2 * x * Subs(Derivative(f(x), x, x), x, x ** 2 + 1)
    e3 = Subs(Derivative(f(x) + y ** 2 - y, y), y, y ** 2)
    assert e3.diff(y) == 4 * y
    e4 = Subs(Derivative(f(x + y), y), y, x ** 2)
    assert e4.diff(y) is S.Zero
    e5 = Subs(Derivative(f(x), x), (y, z), (y, z))
    assert e5.diff(x) == Derivative(f(x), x, x)
    assert f(g(x)).diff(g(x), g(x)) == Derivative(f(g(x)), g(x), g(x))