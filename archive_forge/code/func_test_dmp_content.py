from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dmp_content():
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_content(-2) == 2
    f, g, F = (3 * y ** 2 + 2 * y + 1, 1, 0)
    for i in range(0, 5):
        g *= f
        F += x ** i * g
    assert R.dmp_content(F) == f.drop(x)
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.dmp_content(f_4) == 1
    assert R.dmp_content(f_5) == 1
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    assert R.dmp_content(f_6) == 1