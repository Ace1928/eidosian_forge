from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_logic_fromstring():
    S = Logic.fromstring
    assert S('a') == 'a'
    assert S('!a') == Not('a')
    assert S('a & b') == And('a', 'b')
    assert S('a | b') == Or('a', 'b')
    assert S('a | b & c') == And(Or('a', 'b'), 'c')
    assert S('a & b | c') == Or(And('a', 'b'), 'c')
    assert S('a & b & c') == And('a', 'b', 'c')
    assert S('a | b | c') == Or('a', 'b', 'c')
    raises(ValueError, lambda: S('| a'))
    raises(ValueError, lambda: S('& a'))
    raises(ValueError, lambda: S('a | | b'))
    raises(ValueError, lambda: S('a | & b'))
    raises(ValueError, lambda: S('a & & b'))
    raises(ValueError, lambda: S('a |'))
    raises(ValueError, lambda: S('a|b'))
    raises(ValueError, lambda: S('!'))
    raises(ValueError, lambda: S('! a'))
    raises(ValueError, lambda: S('!(a + 1)'))
    raises(ValueError, lambda: S(''))