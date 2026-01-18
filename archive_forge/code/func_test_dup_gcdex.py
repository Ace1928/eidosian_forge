from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_gcdex():
    R, x = ring('x', QQ)
    f = x ** 4 - 2 * x ** 3 - 6 * x ** 2 + 12 * x + 15
    g = x ** 3 + x ** 2 - 4 * x - 4
    s = -QQ(1, 5) * x + QQ(3, 5)
    t = QQ(1, 5) * x ** 2 - QQ(6, 5) * x + 2
    h = x + 1
    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)
    f = x ** 4 + 4 * x ** 3 - x + 1
    g = x ** 3 - x + 1
    s, t, h = R.dup_gcdex(f, g)
    S, T, H = R.dup_gcdex(g, f)
    assert R.dup_add(R.dup_mul(s, f), R.dup_mul(t, g)) == h
    assert R.dup_add(R.dup_mul(S, g), R.dup_mul(T, f)) == H
    f = 2 * x
    g = x ** 2 - 16
    s = QQ(1, 32) * x
    t = -QQ(1, 16)
    h = 1
    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)