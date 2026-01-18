from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
from sympy.functions.elementary.miscellaneous import sqrt
def test_modgcd_multivariate_integers():
    R, x, y = ring('x,y', ZZ)
    f, g = (R.zero, R.zero)
    assert modgcd_multivariate(f, g) == (0, 0, 0)
    f, g = (2 * x ** 2 + 4 * x + 2, x + 1)
    assert modgcd_multivariate(f, g) == (x + 1, 2 * x + 2, 1)
    f, g = (x + 1, 2 * x ** 2 + 4 * x + 2)
    assert modgcd_multivariate(f, g) == (x + 1, 1, 2 * x + 2)
    f = 2 * x ** 2 + 2 * x * y - 3 * x - 3 * y
    g = 4 * x * y - 2 * x + 4 * y ** 2 - 2 * y
    assert modgcd_multivariate(f, g) == (x + y, 2 * x - 3, 4 * y - 2)
    f, g = (x * y ** 2 + 2 * x * y + x, x * y ** 3 + x)
    assert modgcd_multivariate(f, g) == (x * y + x, y + 1, y ** 2 - y + 1)
    f, g = (x ** 2 * y ** 2 + x ** 2 * y + 1, x * y ** 2 + x * y + 1)
    assert modgcd_multivariate(f, g) == (1, f, g)
    f = x ** 4 + 8 * x ** 3 + 21 * x ** 2 + 22 * x + 8
    g = x ** 3 + 6 * x ** 2 + 11 * x + 6
    h = x ** 2 + 3 * x + 2
    cff = x ** 2 + 5 * x + 4
    cfg = x + 3
    assert modgcd_multivariate(f, g) == (h, cff, cfg)
    R, x, y, z, u = ring('x,y,z,u', ZZ)
    f, g = (x + y + z, -x - y - z - u)
    assert modgcd_multivariate(f, g) == (1, f, g)
    f, g = (u ** 2 + 2 * u + 1, 2 * u + 2)
    assert modgcd_multivariate(f, g) == (u + 1, u + 1, 2)
    f, g = (z ** 2 * u ** 2 + 2 * z ** 2 * u + z ** 2 + z * u + z, u ** 2 + 2 * u + 1)
    h, cff, cfg = (u + 1, z ** 2 * u + z ** 2 + z, u + 1)
    assert modgcd_multivariate(f, g) == (h, cff, cfg)
    assert modgcd_multivariate(g, f) == (h, cfg, cff)
    R, x, y, z = ring('x,y,z', ZZ)
    f, g = (x - y * z, x - y * z)
    assert modgcd_multivariate(f, g) == (x - y * z, 1, 1)
    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)
    R, x, y, z, u, v = ring('x,y,z,u,v', ZZ)
    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)
    R, x, y, z, u, v, a, b = ring('x,y,z,u,v,a,b', ZZ)
    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)
    R, x, y, z, u, v, a, b, c, d = ring('x,y,z,u,v,a,b,c,d', ZZ)
    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)
    R, x, y, z = ring('x,y,z', ZZ)
    f, g, h = R.fateman_poly_F_2()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)
    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = modgcd_multivariate(f, g)
    assert H == h and H * cff == f and (H * cfg == g)