from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
from sympy.functions.elementary.miscellaneous import sqrt
def test_chinese_remainder():
    R, x, y = ring('x, y', ZZ)
    p, q = (3, 5)
    hp = x ** 3 * y - x ** 2 - 1
    hq = -x ** 3 * y - 2 * x * y ** 2 + 2
    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    assert hpq.trunc_ground(p) == hp
    assert hpq.trunc_ground(q) == hq
    T, z = ring('z', R)
    p, q = (3, 7)
    hp = (x * y + 1) * z ** 2 + x
    hq = (x ** 2 - 3 * y) * z + 2
    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    assert hpq.trunc_ground(p) == hp
    assert hpq.trunc_ground(q) == hq