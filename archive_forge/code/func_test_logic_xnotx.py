from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_logic_xnotx():
    assert And('a', Not('a')) == F
    assert Or('a', Not('a')) == T