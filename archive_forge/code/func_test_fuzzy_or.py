from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_fuzzy_or():
    assert fuzzy_or([T, T]) == T
    assert fuzzy_or([T, F]) == T
    assert fuzzy_or([T, U]) == T
    assert fuzzy_or([F, F]) == F
    assert fuzzy_or([F, U]) == U
    assert fuzzy_or([U, U]) == U
    assert [fuzzy_or([w]) for w in [U, T, F]] == [U, T, F]
    assert fuzzy_or([T, F, U]) == T
    assert fuzzy_or([]) == F
    raises(TypeError, lambda: fuzzy_or())