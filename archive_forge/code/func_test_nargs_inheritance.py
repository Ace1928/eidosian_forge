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
def test_nargs_inheritance():

    class f1(Function):
        nargs = 2

    class f2(f1):
        pass

    class f3(f2):
        pass

    class f4(f3):
        nargs = (1, 2)

    class f5(f4):
        pass

    class f6(f5):
        pass

    class f7(f6):
        nargs = None

    class f8(f7):
        pass

    class f9(f8):
        pass

    class f10(f9):
        nargs = 1

    class f11(f10):
        pass
    assert f1.nargs == FiniteSet(2)
    assert f2.nargs == FiniteSet(2)
    assert f3.nargs == FiniteSet(2)
    assert f4.nargs == FiniteSet(1, 2)
    assert f5.nargs == FiniteSet(1, 2)
    assert f6.nargs == FiniteSet(1, 2)
    assert f7.nargs == S.Naturals0
    assert f8.nargs == S.Naturals0
    assert f9.nargs == S.Naturals0
    assert f10.nargs == FiniteSet(1)
    assert f11.nargs == FiniteSet(1)