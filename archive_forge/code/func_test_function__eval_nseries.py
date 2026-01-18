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
def test_function__eval_nseries():
    n = Symbol('n')
    assert sin(x)._eval_nseries(x, 2, None) == x + O(x ** 2)
    assert sin(x + 1)._eval_nseries(x, 2, None) == x * cos(1) + sin(1) + O(x ** 2)
    assert sin(pi * (1 - x))._eval_nseries(x, 2, None) == pi * x + O(x ** 2)
    assert acos(1 - x ** 2)._eval_nseries(x, 2, None) == sqrt(2) * sqrt(x ** 2) + O(x ** 2)
    assert polygamma(n, x + 1)._eval_nseries(x, 2, None) == polygamma(n, 1) + polygamma(n + 1, 1) * x + O(x ** 2)
    raises(PoleError, lambda: sin(1 / x)._eval_nseries(x, 2, None))
    assert acos(1 - x)._eval_nseries(x, 2, None) == sqrt(2) * sqrt(x) + sqrt(2) * x ** (S(3) / 2) / 12 + O(x ** 2)
    assert acos(1 + x)._eval_nseries(x, 2, None) == sqrt(2) * sqrt(-x) + sqrt(2) * (-x) ** (S(3) / 2) / 12 + O(x ** 2)
    assert loggamma(1 / x)._eval_nseries(x, 0, None) == log(x) / 2 - log(x) / x - 1 / x + O(1, x)
    assert loggamma(log(1 / x)).nseries(x, n=1, logx=y) == loggamma(-y)
    assert expint(Rational(3, 2), -x)._eval_nseries(x, 5, None) == 2 - 2 * sqrt(pi) * sqrt(-x) - 2 * x + x ** 2 + x ** 3 / 3 + x ** 4 / 12 + 4 * I * x ** (S(3) / 2) * sqrt(-x) / 3 + 2 * I * x ** (S(5) / 2) * sqrt(-x) / 5 + 2 * I * x ** (S(7) / 2) * sqrt(-x) / 21 + O(x ** 5)
    assert sin(sqrt(x))._eval_nseries(x, 3, None) == sqrt(x) - x ** Rational(3, 2) / 6 + x ** Rational(5, 2) / 120 + O(x ** 3)
    s1 = f(x, y).series(y, n=2)
    assert {i.name for i in s1.atoms(Symbol)} == {'x', 'xi', 'y'}
    xi = Symbol('xi')
    s2 = f(xi, y).series(y, n=2)
    assert {i.name for i in s2.atoms(Symbol)} == {'xi', 'xi0', 'y'}