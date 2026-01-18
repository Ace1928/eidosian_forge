from sympy.polys.orderings import (
from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises
def test_InverseOrder():
    ilex = InverseOrder(lex)
    igrlex = InverseOrder(grlex)
    assert ilex((1, 2, 3)) > ilex((2, 0, 3))
    assert igrlex((1, 2, 3)) < igrlex((0, 2, 3))
    assert str(ilex) == 'ilex'
    assert str(igrlex) == 'igrlex'
    assert ilex.is_global is False
    assert igrlex.is_global is False
    assert ilex != igrlex
    assert ilex == InverseOrder(LexOrder())