from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_ext_rep():
    assert (x ** 2 * z ** (-2) * x).ext_rep == (Symbol('x'), 2, Symbol('z'), -2, Symbol('x'), 1)
    assert (x ** (-2) * y ** (-1)).ext_rep == (Symbol('x'), -2, Symbol('y'), -1)
    assert (x * z).ext_rep == (Symbol('x'), 1, Symbol('z'), 1)