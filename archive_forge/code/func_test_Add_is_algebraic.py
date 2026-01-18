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
def test_Add_is_algebraic():
    a = Symbol('a', algebraic=True)
    b = Symbol('a', algebraic=True)
    na = Symbol('na', algebraic=False)
    nb = Symbol('nb', algebraic=False)
    x = Symbol('x')
    assert (a + b).is_algebraic
    assert (na + nb).is_algebraic is None
    assert (a + na).is_algebraic is False
    assert (a + x).is_algebraic is None
    assert (na + x).is_algebraic is None