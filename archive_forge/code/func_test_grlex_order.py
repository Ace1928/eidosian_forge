from sympy.polys.orderings import (
from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises
def test_grlex_order():
    assert grlex((1, 2, 3)) == (6, (1, 2, 3))
    assert str(grlex) == 'grlex'
    assert grlex((1, 2, 3)) == grlex((1, 2, 3))
    assert grlex((2, 2, 3)) > grlex((1, 2, 3))
    assert grlex((1, 3, 3)) > grlex((1, 2, 3))
    assert grlex((1, 2, 4)) > grlex((1, 2, 3))
    assert grlex((0, 2, 3)) < grlex((1, 2, 3))
    assert grlex((1, 1, 3)) < grlex((1, 2, 3))
    assert grlex((1, 2, 2)) < grlex((1, 2, 3))
    assert grlex((2, 2, 3)) > grlex((1, 2, 4))
    assert grlex((1, 3, 3)) > grlex((1, 2, 4))
    assert grlex((0, 2, 3)) < grlex((1, 2, 2))
    assert grlex((1, 1, 3)) < grlex((1, 2, 2))
    assert grlex((0, 1, 1)) > grlex((0, 0, 2))
    assert grlex((0, 3, 1)) < grlex((2, 2, 1))
    assert grlex.is_global is True