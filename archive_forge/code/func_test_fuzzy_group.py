from sympy.core.logic import (fuzzy_not, Logic, And, Or, Not, fuzzy_and,
from sympy.testing.pytest import raises
from itertools import product
def test_fuzzy_group():
    v = [T, F, U]
    for i in product(*[v] * 3):
        assert _fuzzy_group(i) is (None if None in i else True if all((j for j in i)) else False)
        assert _fuzzy_group(i, quick_exit=True) is (None if i.count(False) > 1 else None if None in i else True if all((j for j in i)) else False)
    it = (True if i == 0 else None for i in range(2))
    assert _torf(it) is None
    it = (True if i == 1 else None for i in range(2))
    assert _torf(it) is None