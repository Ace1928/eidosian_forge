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
def test_common_assumptions():
    assert common_assumptions([0, 1, 2]) == {'algebraic': True, 'irrational': False, 'hermitian': True, 'extended_real': True, 'real': True, 'extended_negative': False, 'extended_nonnegative': True, 'integer': True, 'rational': True, 'imaginary': False, 'complex': True, 'commutative': True, 'noninteger': False, 'composite': False, 'infinite': False, 'nonnegative': True, 'finite': True, 'transcendental': False, 'negative': False}
    assert common_assumptions([0, 1, 2], 'positive integer'.split()) == {'integer': True}
    assert common_assumptions([0, 1, 2], []) == {}
    assert common_assumptions([], ['integer']) == {}
    assert common_assumptions([0], ['integer']) == {'integer': True}