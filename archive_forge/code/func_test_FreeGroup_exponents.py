from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroup_exponents():
    w1 = x ** 2 * y ** 3
    assert w1.exponent_sum(x) == 2
    assert w1.exponent_sum(x ** (-1)) == -2
    assert w1.generator_count(x) == 2
    w2 = x ** 2 * y ** 4 * x ** (-3)
    assert w2.exponent_sum(x) == -1
    assert w2.generator_count(x) == 5