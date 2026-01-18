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
def test_hash_vs_eq():
    """catch: different hash for equal objects"""
    a = 1 + S.Pi
    ha = hash(a)
    a.is_positive
    assert a.is_positive is True
    assert ha == hash(a)
    b = a.expand(trig=True)
    hb = hash(b)
    assert a == b
    assert ha == hb