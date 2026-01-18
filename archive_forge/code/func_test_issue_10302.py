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
def test_issue_10302():
    x = Symbol('x')
    r = Symbol('r', real=True)
    u = -(3 * 2 ** pi) ** (1 / pi) + 2 * 3 ** (1 / pi)
    i = u + u * I
    assert i.is_real is None
    assert (u + i).is_zero is None
    assert (1 + i).is_zero is False
    a = Dummy('a', zero=True)
    assert (a + I).is_zero is False
    assert (a + r * I).is_zero is None
    assert (a + I).is_imaginary
    assert (a + x + I).is_imaginary is None
    assert (a + r * I + I).is_imaginary is None