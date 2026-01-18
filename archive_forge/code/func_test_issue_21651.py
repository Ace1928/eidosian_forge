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
def test_issue_21651():
    k = Symbol('k', positive=True, integer=True)
    exp = 2 * 2 ** (-k)
    assert exp.is_integer is None