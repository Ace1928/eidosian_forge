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
def test_Pow_is_algebraic():
    e = Symbol('e', algebraic=True)
    assert Pow(1, e, evaluate=False).is_algebraic
    assert Pow(0, e, evaluate=False).is_algebraic
    a = Symbol('a', algebraic=True)
    azf = Symbol('azf', algebraic=True, zero=False)
    na = Symbol('na', algebraic=False)
    ia = Symbol('ia', algebraic=True, irrational=True)
    ib = Symbol('ib', algebraic=True, irrational=True)
    r = Symbol('r', rational=True)
    x = Symbol('x')
    assert (a ** 2).is_algebraic is True
    assert (a ** r).is_algebraic is None
    assert (azf ** r).is_algebraic is True
    assert (a ** x).is_algebraic is None
    assert (na ** r).is_algebraic is None
    assert (ia ** r).is_algebraic is True
    assert (ia ** ib).is_algebraic is False
    assert (a ** e).is_algebraic is None
    assert Pow(2, sqrt(2), evaluate=False).is_algebraic is False
    assert Pow(S.GoldenRatio, sqrt(3), evaluate=False).is_algebraic is False
    t = Symbol('t', real=True, transcendental=True)
    n = Symbol('n', integer=True)
    assert (t ** n).is_algebraic is None
    assert (t ** n).is_integer is None
    assert (pi ** 3).is_algebraic is False
    r = Symbol('r', zero=True)
    assert (pi ** r).is_algebraic is True