from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_groebner_gcd():
    R, x, y, z = ring('x,y,z', ZZ)
    assert groebner_gcd(x ** 2 - y ** 2, x - y) == x - y
    assert groebner_gcd(2 * x ** 2 - 2 * y ** 2, 2 * x - 2 * y) == 2 * x - 2 * y
    R, x, y, z = ring('x,y,z', QQ)
    assert groebner_gcd(x ** 2 - y ** 2, x - y) == x - y
    assert groebner_gcd(2 * x ** 2 - 2 * y ** 2, 2 * x - 2 * y) == x - y