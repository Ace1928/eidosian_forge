from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_logic_onearg():
    assert And() is True
    assert Or() is False
    assert And(T) == T
    assert And(F) == F
    assert Or(T) == T
    assert Or(F) == F
    assert And('a') == 'a'
    assert Or('a') == 'a'