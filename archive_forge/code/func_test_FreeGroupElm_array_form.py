from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_array_form():
    assert (x * z).array_form == ((Symbol('x'), 1), (Symbol('z'), 1))
    assert (x ** 2 * z * y * x ** (-2)).array_form == ((Symbol('x'), 2), (Symbol('z'), 1), (Symbol('y'), 1), (Symbol('x'), -2))
    assert (x ** (-2) * y ** (-1)).array_form == ((Symbol('x'), -2), (Symbol('y'), -1))