from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dmp_discriminant():
    R, x = ring('x', ZZ)
    assert R.dmp_discriminant(0) == 0
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_discriminant(0) == 0
    assert R.dmp_discriminant(y) == 0
    assert R.dmp_discriminant(x ** 3 + 3 * x ** 2 + 9 * x - 13) == -11664
    assert R.dmp_discriminant(5 * x ** 5 + x ** 3 + 2) == 31252160
    assert R.dmp_discriminant(x ** 4 + 2 * x ** 3 + 6 * x ** 2 - 22 * x + 13) == 0
    assert R.dmp_discriminant(12 * x ** 7 + 15 * x ** 4 + 30 * x ** 3 + x ** 2 + 1) == -220289699947514112
    assert R.dmp_discriminant(x ** 2 * y + 2 * y) == (-8 * y ** 2).drop(x)
    assert R.dmp_discriminant(x * y ** 2 + 2 * x) == 1
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.dmp_discriminant(x * y + z) == 1
    R, x, y, z, u = ring('x,y,z,u', ZZ)
    assert R.dmp_discriminant(x ** 2 * y + x * z + u) == (-4 * y * u + z ** 2).drop(x)
    R, x, y, z, u, v = ring('x,y,z,u,v', ZZ)
    assert R.dmp_discriminant(x ** 3 * y + x ** 2 * z + x * u + v) == (-27 * y ** 2 * v ** 2 + 18 * y * z * u * v - 4 * y * u ** 3 - 4 * z ** 3 * v + z ** 2 * u ** 2).drop(x)