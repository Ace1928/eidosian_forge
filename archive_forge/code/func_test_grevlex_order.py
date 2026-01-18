from sympy.polys.orderings import (
from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises
def test_grevlex_order():
    assert grevlex((1, 2, 3)) == (6, (-3, -2, -1))
    assert str(grevlex) == 'grevlex'
    assert grevlex((1, 2, 3)) == grevlex((1, 2, 3))
    assert grevlex((2, 2, 3)) > grevlex((1, 2, 3))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 3))
    assert grevlex((1, 2, 4)) > grevlex((1, 2, 3))
    assert grevlex((0, 2, 3)) < grevlex((1, 2, 3))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 3))
    assert grevlex((1, 2, 2)) < grevlex((1, 2, 3))
    assert grevlex((2, 2, 3)) > grevlex((1, 2, 4))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 4))
    assert grevlex((0, 2, 3)) < grevlex((1, 2, 2))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 2))
    assert grevlex((0, 1, 1)) > grevlex((0, 0, 2))
    assert grevlex((0, 3, 1)) < grevlex((2, 2, 1))
    assert grevlex.is_global is True