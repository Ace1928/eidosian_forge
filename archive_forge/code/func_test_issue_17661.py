from itertools import permutations
from sympy.core.expr import unchanged
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.combinatorics.permutations import \
from sympy.printing import sstr, srepr, pretty, latex
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_issue_17661():
    c1 = Cycle(1, 2)
    c2 = Cycle(1, 2)
    assert c1 == c2
    assert repr(c1) == 'Cycle(1, 2)'
    assert c1 == c2