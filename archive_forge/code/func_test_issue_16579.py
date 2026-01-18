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
def test_issue_16579():
    x = Symbol('x', extended_real=True, infinite=False)
    y = Symbol('y', extended_real=True, finite=False)
    assert x.is_finite is True
    assert y.is_infinite is True
    c = Symbol('c', complex=True)
    assert c.is_finite is True
    raises(InconsistentAssumptions, lambda: Dummy(complex=True, finite=False))
    nf = Symbol('nf', finite=False)
    assert nf.is_infinite is True