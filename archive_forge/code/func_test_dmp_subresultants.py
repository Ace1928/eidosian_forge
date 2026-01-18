from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dmp_subresultants():
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_resultant(0, 0) == 0
    assert R.dmp_prs_resultant(0, 0)[0] == 0
    assert R.dmp_zz_collins_resultant(0, 0) == 0
    assert R.dmp_qq_collins_resultant(0, 0) == 0
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(0, 1) == 0
    assert R.dmp_prs_resultant(0, 1)[0] == 0
    assert R.dmp_zz_collins_resultant(0, 1) == 0
    assert R.dmp_qq_collins_resultant(0, 1) == 0
    f = 3 * x ** 2 * y - y ** 3 - 4
    g = x ** 2 + x * y ** 3 - 9
    a = 3 * x * y ** 4 + y ** 3 - 27 * y + 4
    b = -3 * y ** 10 - 12 * y ** 7 + y ** 6 - 54 * y ** 4 + 8 * y ** 3 + 729 * y ** 2 - 216 * y + 16
    r = R.dmp_LC(b)
    assert R.dmp_subresultants(f, g) == [f, g, a, b]
    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r
    f = -x ** 3 + 5
    g = 3 * x ** 2 * y + x ** 2
    a = 45 * y ** 2 + 30 * y + 5
    b = 675 * y ** 3 + 675 * y ** 2 + 225 * y + 25
    r = R.dmp_LC(b)
    assert R.dmp_subresultants(f, g) == [f, g, a]
    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r
    R, x, y, z, u, v = ring('x,y,z,u,v', ZZ)
    f = 6 * x ** 2 - 3 * x * y - 2 * x * z + y * z
    g = x ** 2 - x * u - x * v + u * v
    r = y ** 2 * z ** 2 - 3 * y ** 2 * z * u - 3 * y ** 2 * z * v + 9 * y ** 2 * u * v - 2 * y * z ** 2 * u - 2 * y * z ** 2 * v + 6 * y * z * u ** 2 + 12 * y * z * u * v + 6 * y * z * v ** 2 - 18 * y * u ** 2 * v - 18 * y * u * v ** 2 + 4 * z ** 2 * u * v - 12 * z * u ** 2 * v - 12 * z * u * v ** 2 + 36 * u ** 2 * v ** 2
    assert R.dmp_zz_collins_resultant(f, g) == r.drop(x)
    R, x, y, z, u, v = ring('x,y,z,u,v', QQ)
    f = x ** 2 - QQ(1, 2) * x * y - QQ(1, 3) * x * z + QQ(1, 6) * y * z
    g = x ** 2 - x * u - x * v + u * v
    r = QQ(1, 36) * y ** 2 * z ** 2 - QQ(1, 12) * y ** 2 * z * u - QQ(1, 12) * y ** 2 * z * v + QQ(1, 4) * y ** 2 * u * v - QQ(1, 18) * y * z ** 2 * u - QQ(1, 18) * y * z ** 2 * v + QQ(1, 6) * y * z * u ** 2 + QQ(1, 3) * y * z * u * v + QQ(1, 6) * y * z * v ** 2 - QQ(1, 2) * y * u ** 2 * v - QQ(1, 2) * y * u * v ** 2 + QQ(1, 9) * z ** 2 * u * v - QQ(1, 3) * z * u ** 2 * v - QQ(1, 3) * z * u * v ** 2 + u ** 2 * v ** 2
    assert R.dmp_qq_collins_resultant(f, g) == r.drop(x)
    Rt, t = ring('t', ZZ)
    Rx, x = ring('x', Rt)
    f = x ** 6 - 5 * x ** 4 + 5 * x ** 2 + 4
    g = -6 * t * x ** 5 + x ** 4 + 20 * t * x ** 3 - 3 * x ** 2 - 10 * t * x + 6
    assert Rx.dup_resultant(f, g) == 2930944 * t ** 6 + 2198208 * t ** 4 + 549552 * t ** 2 + 45796