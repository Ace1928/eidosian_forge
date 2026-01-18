from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_f5_reduce():
    R, x, y, z = ring('x,y,z', QQ, lex)
    F = [(((0, 0, 0), 1), x + 2 * y + 2 * z - 1, 1), (((0, 0, 0), 2), 6 * y ** 2 + 8 * y * z - 2 * y + 6 * z ** 2 - 2 * z, 2), (((0, 0, 0), 3), QQ(10, 3) * y * z - QQ(1, 3) * y + 4 * z ** 2 - QQ(4, 3) * z, 3), (((0, 0, 1), 2), y + 30 * z ** 3 - QQ(79, 7) * z ** 2 + QQ(3, 7) * z, 4), (((0, 0, 2), 2), z ** 4 - QQ(10, 21) * z ** 3 + QQ(1, 84) * z ** 2 + QQ(1, 84) * z, 5)]
    cp = critical_pair(F[0], F[1], R)
    s = s_poly(cp)
    assert f5_reduce(s, F) == (((0, 2, 0), 1), R.zero, 1)
    s = lbp(sig(Sign(s)[0], 100), Polyn(s), Num(s))
    assert f5_reduce(s, F) == s