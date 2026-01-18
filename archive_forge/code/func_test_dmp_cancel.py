from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dmp_cancel():
    R, x, y = ring('x,y', ZZ)
    f = 2 * x ** 2 - 2
    g = x ** 2 - 2 * x + 1
    p = 2 * x + 2
    q = x - 1
    assert R.dmp_cancel(f, g) == (p, q)
    assert R.dmp_cancel(f, g, include=False) == (1, 1, p, q)
    assert R.dmp_cancel(0, 0) == (0, 0)
    assert R.dmp_cancel(0, 0, include=False) == (1, 1, 0, 0)
    assert R.dmp_cancel(y, 0) == (1, 0)
    assert R.dmp_cancel(y, 0, include=False) == (1, 1, 1, 0)
    assert R.dmp_cancel(0, y) == (0, 1)
    assert R.dmp_cancel(0, y, include=False) == (1, 1, 0, 1)