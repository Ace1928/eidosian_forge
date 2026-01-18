from sympy.combinatorics.free_groups import free_group, FreeGroup
from sympy.core import Symbol
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_FreeGroupElm_type_error():
    raises(TypeError, lambda: 2 / x)
    raises(TypeError, lambda: x ** 2 + y ** 2)
    raises(TypeError, lambda: x / 2)