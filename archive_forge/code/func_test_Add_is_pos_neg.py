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
def test_Add_is_pos_neg():
    n = Symbol('n', extended_negative=True, infinite=True)
    nn = Symbol('n', extended_nonnegative=True, infinite=True)
    np = Symbol('n', extended_nonpositive=True, infinite=True)
    p = Symbol('p', extended_positive=True, infinite=True)
    r = Dummy(extended_real=True, finite=False)
    x = Symbol('x')
    xf = Symbol('xf', finite=True)
    assert (n + p).is_extended_positive is None
    assert (n + x).is_extended_positive is None
    assert (p + x).is_extended_positive is None
    assert (n + p).is_extended_negative is None
    assert (n + x).is_extended_negative is None
    assert (p + x).is_extended_negative is None
    assert (n + xf).is_extended_positive is False
    assert (p + xf).is_extended_positive is True
    assert (n + xf).is_extended_negative is True
    assert (p + xf).is_extended_negative is False
    assert (x - S.Infinity).is_extended_negative is None
    assert (p + nn).is_extended_positive
    assert (n + np).is_extended_negative
    assert (p + r).is_extended_positive is None