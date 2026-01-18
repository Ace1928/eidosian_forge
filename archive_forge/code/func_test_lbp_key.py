from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_lbp_key():
    R, x, y, z, t = ring('x,y,z,t', ZZ, lex)
    p1 = lbp(sig((0,) * 4, 3), R.zero, 12)
    p2 = lbp(sig((0,) * 4, 4), R.zero, 13)
    p3 = lbp(sig((0,) * 4, 4), R.zero, 12)
    assert lbp_key(p1) > lbp_key(p2)
    assert lbp_key(p2) < lbp_key(p3)