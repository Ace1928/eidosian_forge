from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_subresultants():
    R, x = ring('x', ZZ)
    assert R.dup_resultant(0, 0) == 0
    assert R.dup_resultant(1, 0) == 0
    assert R.dup_resultant(0, 1) == 0
    f = x ** 8 + x ** 6 - 3 * x ** 4 - 3 * x ** 3 + 8 * x ** 2 + 2 * x - 5
    g = 3 * x ** 6 + 5 * x ** 4 - 4 * x ** 2 - 9 * x + 21
    a = 15 * x ** 4 - 3 * x ** 2 + 9
    b = 65 * x ** 2 + 125 * x - 245
    c = 9326 * x - 12300
    d = 260708
    assert R.dup_subresultants(f, g) == [f, g, a, b, c, d]
    assert R.dup_resultant(f, g) == R.dup_LC(d)
    f = x ** 2 - 2 * x + 1
    g = x ** 2 - 1
    a = 2 * x - 2
    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 0
    f = x ** 2 + 1
    g = x ** 2 - 1
    a = -2
    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 4
    f = x ** 2 - 1
    g = x ** 3 - x ** 2 + 2
    assert R.dup_resultant(f, g) == 0
    f = 3 * x ** 3 - x
    g = 5 * x ** 2 + 1
    assert R.dup_resultant(f, g) == 64
    f = x ** 2 - 2 * x + 7
    g = x ** 3 - x + 5
    assert R.dup_resultant(f, g) == 265
    f = x ** 3 - 6 * x ** 2 + 11 * x - 6
    g = x ** 3 - 15 * x ** 2 + 74 * x - 120
    assert R.dup_resultant(f, g) == -8640
    f = x ** 3 - 6 * x ** 2 + 11 * x - 6
    g = x ** 3 - 10 * x ** 2 + 29 * x - 20
    assert R.dup_resultant(f, g) == 0
    f = x ** 3 - 1
    g = x ** 3 + 2 * x ** 2 + 2 * x - 1
    assert R.dup_resultant(f, g) == 16
    f = x ** 8 - 2
    g = x - 1
    assert R.dup_resultant(f, g) == -1