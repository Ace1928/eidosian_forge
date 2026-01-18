from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_words():
    w = x ** 5 * y * x ** 2 * y ** (-4) * x
    assert w.subword(2, 6) == x ** 3 * y
    assert w.subword(3, 2) == F.identity
    assert w.subword(6, 10) == x ** 2 * y ** (-2)
    assert w.substituted_word(0, 7, y ** (-1)) == y ** (-1) * x * y ** (-4) * x
    assert w.substituted_word(0, 7, y ** 2 * x) == y ** 2 * x ** 2 * y ** (-4) * x