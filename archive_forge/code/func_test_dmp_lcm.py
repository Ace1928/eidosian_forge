from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dmp_lcm():
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_lcm(2, 6) == 6
    assert R.dmp_lcm(x, y) == x * y
    assert R.dmp_lcm(2 * x ** 3, 6 * x * y ** 2) == 6 * x ** 3 * y ** 2
    assert R.dmp_lcm(2 * x ** 3, 3 * x * y ** 2) == 6 * x ** 3 * y ** 2
    assert R.dmp_lcm(x ** 2 * y, x * y ** 2) == x ** 2 * y ** 2
    f = 2 * x * y ** 5 - 3 * x * y ** 4 - 2 * x * y ** 3 + 3 * x * y ** 2
    g = y ** 5 - 2 * y ** 3 + y
    h = 2 * x * y ** 7 - 3 * x * y ** 6 - 4 * x * y ** 5 + 6 * x * y ** 4 + 2 * x * y ** 3 - 3 * x * y ** 2
    assert R.dmp_lcm(f, g) == h
    f = x ** 3 - 3 * x ** 2 * y - 9 * x * y ** 2 - 5 * y ** 3
    g = x ** 4 + 6 * x ** 3 * y + 12 * x ** 2 * y ** 2 + 10 * x * y ** 3 + 3 * y ** 4
    h = x ** 5 + x ** 4 * y - 18 * x ** 3 * y ** 2 - 50 * x ** 2 * y ** 3 - 47 * x * y ** 4 - 15 * y ** 5
    assert R.dmp_lcm(f, g) == h