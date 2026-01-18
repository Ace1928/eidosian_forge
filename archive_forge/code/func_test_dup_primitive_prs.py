from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_primitive_prs():
    R, x = ring('x', ZZ)
    f = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    g = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    assert R.dup_primitive_prs(f, g) == [f, g, -5 * x ** 4 + x ** 2 - 3, 13 * x ** 2 + 25 * x - 49, 4663 * x - 6150, 1]