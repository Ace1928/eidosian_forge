from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_fuzzy_and():
    assert fuzzy_and([T, T]) == T
    assert fuzzy_and([T, F]) == F
    assert fuzzy_and([T, U]) == U
    assert fuzzy_and([F, F]) == F
    assert fuzzy_and([F, U]) == F
    assert fuzzy_and([U, U]) == U
    assert [fuzzy_and([w]) for w in [U, T, F]] == [U, T, F]
    assert fuzzy_and([T, F, U]) == F
    assert fuzzy_and([]) == T
    raises(TypeError, lambda: fuzzy_and())