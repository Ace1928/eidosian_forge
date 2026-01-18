from sympy.core.mod import Mod
from sympy.core.numbers import (I, oo, pi)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.simplify.simplify import simplify
from sympy.core import Symbol, S, Rational, Integer, Dummy, Wild, Pow
from sympy.core.assumptions import (assumptions, check_assumptions,
from sympy.core.facts import InconsistentAssumptions
from sympy.core.random import seed
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.testing.pytest import raises, XFAIL
def test_Add_is_infinite():
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    i2 = Symbol('i2', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.add import Add
    assert (x + f).is_finite is None
    assert (x + i).is_finite is None
    assert (f + i).is_finite is False
    assert (x + f + i).is_finite is None
    assert (z + i).is_finite is False
    assert (nzf + i).is_finite is False
    assert (z + f).is_finite is True
    assert (i + i2).is_finite is None
    assert Add(0, f, evaluate=False).is_finite is True
    assert Add(0, i, evaluate=False).is_finite is False
    assert (x + f).is_infinite is None
    assert (x + i).is_infinite is None
    assert (f + i).is_infinite is True
    assert (x + f + i).is_infinite is None
    assert (z + i).is_infinite is True
    assert (nzf + i).is_infinite is True
    assert (z + f).is_infinite is False
    assert (i + i2).is_infinite is None
    assert Add(0, f, evaluate=False).is_infinite is False
    assert Add(0, i, evaluate=False).is_infinite is True