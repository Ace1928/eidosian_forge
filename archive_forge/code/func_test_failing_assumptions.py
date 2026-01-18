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
def test_failing_assumptions():
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert failing_assumptions(6 * x + y, **x.assumptions0) == {'real': None, 'imaginary': None, 'complex': None, 'hermitian': None, 'positive': None, 'nonpositive': None, 'nonnegative': None, 'nonzero': None, 'negative': None, 'zero': None, 'extended_real': None, 'finite': None, 'infinite': None, 'extended_negative': None, 'extended_nonnegative': None, 'extended_nonpositive': None, 'extended_nonzero': None, 'extended_positive': None}