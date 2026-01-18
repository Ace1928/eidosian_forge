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
def test_Subs():
    assert Subs(1, (), ()) is S.One
    assert Subs(x, y, z) != Subs(x, y, 1)
    assert Subs(x, x, 1).subs(x, y).has(y)
    assert Subs(Derivative(f(x), (x, 2)), x, x).doit() == f(x).diff(x, x)
    assert Subs(x, x, 0).has(x)
    assert not Subs(x, x, 0).free_symbols
    assert Subs(Subs(x + y, x, 2), y, 1) == Subs(x + y, (x, y), (2, 1))
    assert Subs(x, (x,), (0,)) == Subs(x, x, 0)
    assert Subs(x, x, 0) == Subs(y, y, 0)
    assert Subs(x, x, 0).subs(x, 1) == Subs(x, x, 0)
    assert Subs(y, x, 0).subs(y, 1) == Subs(1, x, 0)
    assert Subs(f(x), x, 0).doit() == f(0)
    assert Subs(f(x ** 2), x ** 2, 0).doit() == f(0)
    assert Subs(f(x, y, z), (x, y, z), (0, 1, 1)) != Subs(f(x, y, z), (x, y, z), (0, 0, 1))
    assert Subs(x, y, 2).subs(x, y).doit() == 2
    assert Subs(f(x, y), (x, y, z), (0, 1, 1)) != Subs(f(x, y) + z, (x, y, z), (0, 1, 0))
    assert Subs(f(x, y), (x, y), (0, 1)).doit() == f(0, 1)
    assert Subs(Subs(f(x, y), x, 0), y, 1).doit() == f(0, 1)
    raises(ValueError, lambda: Subs(f(x, y), (x, y), (0, 0, 1)))
    raises(ValueError, lambda: Subs(f(x, y), (x, x, y), (0, 0, 1)))
    assert len(Subs(f(x, y), (x, y), (0, 1)).variables) == 2
    assert Subs(f(x, y), (x, y), (0, 1)).point == Tuple(0, 1)
    assert Subs(f(x), x, 0) == Subs(f(y), y, 0)
    assert Subs(f(x, y), (x, y), (0, 1)) == Subs(f(x, y), (y, x), (1, 0))
    assert Subs(f(x) * y, (x, y), (0, 1)) == Subs(f(y) * x, (y, x), (0, 1))
    assert Subs(f(x) * y, (x, y), (1, 1)) == Subs(f(y) * x, (x, y), (1, 1))
    assert Subs(f(x), x, 0).subs(x, 1).doit() == f(0)
    assert Subs(f(x), x, y).subs(y, 0) == Subs(f(x), x, 0)
    assert Subs(y * f(x), x, y).subs(y, 2) == Subs(2 * f(x), x, 2)
    assert (2 * Subs(f(x), x, 0)).subs(Subs(f(x), x, 0), y) == 2 * y
    assert Subs(f(x), x, 0).free_symbols == set()
    assert Subs(f(x, y), x, z).free_symbols == {y, z}
    assert Subs(f(x).diff(x), x, 0).doit(), Subs(f(x).diff(x), x, 0)
    assert Subs(1 + f(x).diff(x), x, 0).doit(), 1 + Subs(f(x).diff(x), x, 0)
    assert Subs(y * f(x, y).diff(x), (x, y), (0, 2)).doit() == 2 * Subs(Derivative(f(x, 2), x), x, 0)
    assert Subs(y ** 2 * f(x), x, 0).diff(y) == 2 * y * f(0)
    e = Subs(y ** 2 * f(x), x, y)
    assert e.diff(y) == e.doit().diff(y) == y ** 2 * Derivative(f(y), y) + 2 * y * f(y)
    assert Subs(f(x), x, 0) + Subs(f(x), x, 0) == 2 * Subs(f(x), x, 0)
    e1 = Subs(z * f(x), x, 1)
    e2 = Subs(z * f(y), y, 1)
    assert e1 + e2 == 2 * e1
    assert e1.__hash__() == e2.__hash__()
    assert Subs(z * f(x + 1), x, 1) not in [e1, e2]
    assert Derivative(f(x), x).subs(x, g(x)) == Derivative(f(g(x)), g(x))
    assert Derivative(f(x), x).subs(x, x + y) == Subs(Derivative(f(x), x), x, x + y)
    assert Subs(f(x) * cos(y) + z, (x, y), (0, pi / 3)).n(2) == Subs(f(x) * cos(y) + z, (x, y), (0, pi / 3)).evalf(2) == z + Rational('1/2').n(2) * f(0)
    assert f(x).diff(x).subs(x, 0).subs(x, y) == f(x).diff(x).subs(x, 0)
    assert (x * f(x).diff(x).subs(x, 0)).subs(x, y) == y * f(x).diff(x).subs(x, 0)
    assert Subs(Derivative(g(x) ** 2, g(x), x), g(x), exp(x)).doit() == 2 * exp(x)
    assert Subs(Derivative(g(x) ** 2, g(x), x), g(x), exp(x)).doit(deep=False) == 2 * Derivative(exp(x), x)
    assert Derivative(f(x, g(x)), x).doit() == Derivative(f(x, g(x)), g(x)) * Derivative(g(x), x) + Subs(Derivative(f(y, g(x)), y), y, x)