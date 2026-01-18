from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_logic_expand():
    t = And(Or('a', 'b'), 'c')
    assert t.expand() == Or(And('a', 'c'), And('b', 'c'))
    t = And(Or('a', Not('b')), 'b')
    assert t.expand() == And('a', 'b')
    t = And(Or('a', 'b'), Or('c', 'd'))
    assert t.expand() == Or(And('a', 'c'), And('a', 'd'), And('b', 'c'), And('b', 'd'))