from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_syllables():
    w = x ** 5 * y * x ** 2 * y ** (-4) * x
    assert w.number_syllables() == 5
    assert w.exponent_syllable(2) == 2
    assert w.generator_syllable(3) == Symbol('y')
    assert w.sub_syllables(1, 2) == y
    assert w.sub_syllables(3, 3) == F.identity