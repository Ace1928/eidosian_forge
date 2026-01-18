from sympy.polys.orderings import (
from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises
def test_ProductOrder():
    P = ProductOrder((grlex, lambda m: m[:2]), (grlex, lambda m: m[2:]))
    assert P((1, 3, 3, 4, 5)) > P((2, 1, 5, 5, 5))
    assert str(P) == 'ProductOrder(grlex, grlex)'
    assert P.is_global is True
    assert ProductOrder((grlex, None), (ilex, None)).is_global is None
    assert ProductOrder((igrlex, None), (ilex, None)).is_global is False