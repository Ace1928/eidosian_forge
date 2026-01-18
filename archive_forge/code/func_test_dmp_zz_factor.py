from sympy.polys.rings import ring, xring
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys
from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises, XFAIL
def test_dmp_zz_factor():
    R, x = ring('x', ZZ)
    assert R.dmp_zz_factor(0) == (0, [])
    assert R.dmp_zz_factor(7) == (7, [])
    assert R.dmp_zz_factor(-7) == (-7, [])
    assert R.dmp_zz_factor(x ** 2 - 9) == (1, [(x - 3, 1), (x + 3, 1)])
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_zz_factor(0) == (0, [])
    assert R.dmp_zz_factor(7) == (7, [])
    assert R.dmp_zz_factor(-7) == (-7, [])
    assert R.dmp_zz_factor(x) == (1, [(x, 1)])
    assert R.dmp_zz_factor(4 * x) == (4, [(x, 1)])
    assert R.dmp_zz_factor(4 * x + 2) == (2, [(2 * x + 1, 1)])
    assert R.dmp_zz_factor(x * y + 1) == (1, [(x * y + 1, 1)])
    assert R.dmp_zz_factor(y ** 2 + 1) == (1, [(y ** 2 + 1, 1)])
    assert R.dmp_zz_factor(y ** 2 - 1) == (1, [(y - 1, 1), (y + 1, 1)])
    assert R.dmp_zz_factor(x ** 2 * y ** 2 + 6 * x ** 2 * y + 9 * x ** 2 - 1) == (1, [(x * y + 3 * x - 1, 1), (x * y + 3 * x + 1, 1)])
    assert R.dmp_zz_factor(x ** 2 * y ** 2 - 9) == (1, [(x * y - 3, 1), (x * y + 3, 1)])
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.dmp_zz_factor(x ** 2 * y ** 2 * z ** 2 - 9) == (1, [(x * y * z - 3, 1), (x * y * z + 3, 1)])
    R, x, y, z, u = ring('x,y,z,u', ZZ)
    assert R.dmp_zz_factor(x ** 2 * y ** 2 * z ** 2 * u ** 2 - 9) == (1, [(x * y * z * u - 3, 1), (x * y * z * u + 3, 1)])
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.dmp_zz_factor(f_1) == (1, [(x + y * z + 20, 1), (x * y + z + 10, 1), (x * z + y + 30, 1)])
    assert R.dmp_zz_factor(f_2) == (1, [(x ** 2 * y ** 2 + x ** 2 * z ** 2 + y + 90, 1), (x ** 3 * y + x ** 3 * z + z - 11, 1)])
    assert R.dmp_zz_factor(f_3) == (1, [(x ** 2 * y ** 2 + x * z ** 4 + x + z, 1), (x ** 3 + x * y * z + y ** 2 + y * z ** 3, 1)])
    assert R.dmp_zz_factor(f_4) == (-1, [(x * y ** 3 + z ** 2, 1), (x ** 2 * z + y ** 4 * z ** 2 + 5, 1), (x ** 3 * y - z ** 2 - 3, 1), (x ** 3 * y ** 4 + z ** 2, 1)])
    assert R.dmp_zz_factor(f_5) == (-1, [(x + y - z, 3)])
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    assert R.dmp_zz_factor(f_6) == (1, [(47 * x * y + z ** 3 * t ** 2 - t ** 2, 1), (45 * x ** 3 - 9 * y ** 3 - y ** 2 + 3 * z ** 3 + 2 * z * t, 1)])
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.dmp_zz_factor(w_1) == (1, [(x ** 2 * y ** 2 - x ** 2 * z ** 2 + y - z ** 2, 1), (x ** 2 * y * z ** 2 + 3 * x * z + 2 * y, 1), (4 * x ** 2 * y + 4 * x ** 2 * z + x * y * z - 1, 1)])
    R, x, y = ring('x,y', ZZ)
    f = -12 * x ** 16 * y + 240 * x ** 12 * y ** 3 - 768 * x ** 10 * y ** 4 + 1080 * x ** 8 * y ** 5 - 768 * x ** 6 * y ** 6 + 240 * x ** 4 * y ** 7 - 12 * y ** 9
    assert R.dmp_zz_factor(f) == (-12, [(y, 1), (x ** 2 - y, 6), (x ** 4 + 6 * x ** 2 * y + y ** 2, 1)])