from sympy.polys.rings import ring
from sympy.polys.domains import ZZ
from sympy.polys.heuristicgcd import heugcd
def test_issue_10996():
    R, x, y, z = ring('x,y,z', ZZ)
    f = 12 * x ** 6 * y ** 7 * z ** 3 - 3 * x ** 4 * y ** 9 * z ** 3 + 12 * x ** 3 * y ** 5 * z ** 4
    g = -48 * x ** 7 * y ** 8 * z ** 3 + 12 * x ** 5 * y ** 10 * z ** 3 - 48 * x ** 5 * y ** 7 * z ** 2 + 36 * x ** 4 * y ** 7 * z - 48 * x ** 4 * y ** 6 * z ** 4 + 12 * x ** 3 * y ** 9 * z ** 2 - 48 * x ** 3 * y ** 4 - 9 * x ** 2 * y ** 9 * z - 48 * x ** 2 * y ** 5 * z ** 3 + 12 * x * y ** 6 + 36 * x * y ** 5 * z ** 2 - 48 * y ** 2 * z
    H, cff, cfg = heugcd(f, g)
    assert H == 12 * x ** 3 * y ** 4 - 3 * x * y ** 6 + 12 * y ** 2 * z
    assert H * cff == f and H * cfg == g