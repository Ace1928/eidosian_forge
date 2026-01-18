from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_euclidean_prs():
    R, x = ring('x', QQ)
    f = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    g = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    assert R.dup_euclidean_prs(f, g) == [f, g, -QQ(5, 9) * x ** 4 + QQ(1, 9) * x ** 2 - QQ(1, 3), -QQ(117, 25) * x ** 2 - 9 * x + QQ(441, 25), QQ(233150, 19773) * x - QQ(102500, 6591), -QQ(1288744821, 543589225)]