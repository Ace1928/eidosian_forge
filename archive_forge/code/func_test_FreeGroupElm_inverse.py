from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_inverse():
    assert x.inverse() == x ** (-1)
    assert (x * y).inverse() == y ** (-1) * x ** (-1)
    assert (y * x * y ** (-1)).inverse() == y * x ** (-1) * y ** (-1)
    assert (y ** 2 * x ** (-1)).inverse() == x * y ** (-2)