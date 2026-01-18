from sympy.polys.orderings import (
from sympy.abc import x, y, z, t
from sympy.core import S
from sympy.testing.pytest import raises
def test_build_product_order():
    assert build_product_order((('grlex', x, y), ('grlex', z, t)), [x, y, z, t])((4, 5, 6, 7)) == ((9, (4, 5)), (13, (6, 7)))
    assert build_product_order((('grlex', x, y), ('grlex', z, t)), [x, y, z, t]) == build_product_order((('grlex', x, y), ('grlex', z, t)), [x, y, z, t])