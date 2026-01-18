from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_cancel():
    R, x = ring('x', ZZ)
    f = 2 * x ** 2 - 2
    g = x ** 2 - 2 * x + 1
    p = 2 * x + 2
    q = x - 1
    assert R.dup_cancel(f, g) == (p, q)
    assert R.dup_cancel(f, g, include=False) == (1, 1, p, q)
    f = -x - 2
    g = 3 * x - 4
    F = x + 2
    G = -3 * x + 4
    assert R.dup_cancel(f, g) == (f, g)
    assert R.dup_cancel(F, G) == (f, g)
    assert R.dup_cancel(0, 0) == (0, 0)
    assert R.dup_cancel(0, 0, include=False) == (1, 1, 0, 0)
    assert R.dup_cancel(x, 0) == (1, 0)
    assert R.dup_cancel(x, 0, include=False) == (1, 1, 1, 0)
    assert R.dup_cancel(0, x) == (0, 1)
    assert R.dup_cancel(0, x, include=False) == (1, 1, 0, 1)
    f = 0
    g = x
    one = 1
    assert R.dup_cancel(f, g, include=True) == (f, one)