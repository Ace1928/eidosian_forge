from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_groebner_lcm():
    R, x, y, z = ring('x,y,z', ZZ)
    assert groebner_lcm(x ** 2 - y ** 2, x - y) == x ** 2 - y ** 2
    assert groebner_lcm(2 * x ** 2 - 2 * y ** 2, 2 * x - 2 * y) == 2 * x ** 2 - 2 * y ** 2
    R, x, y, z = ring('x,y,z', QQ)
    assert groebner_lcm(x ** 2 - y ** 2, x - y) == x ** 2 - y ** 2
    assert groebner_lcm(2 * x ** 2 - 2 * y ** 2, 2 * x - 2 * y) == 2 * x ** 2 - 2 * y ** 2
    R, x, y = ring('x,y', ZZ)
    assert groebner_lcm(x ** 2 * y, x * y ** 2) == x ** 2 * y ** 2
    f = 2 * x * y ** 5 - 3 * x * y ** 4 - 2 * x * y ** 3 + 3 * x * y ** 2
    g = y ** 5 - 2 * y ** 3 + y
    h = 2 * x * y ** 7 - 3 * x * y ** 6 - 4 * x * y ** 5 + 6 * x * y ** 4 + 2 * x * y ** 3 - 3 * x * y ** 2
    assert groebner_lcm(f, g) == h
    f = x ** 3 - 3 * x ** 2 * y - 9 * x * y ** 2 - 5 * y ** 3
    g = x ** 4 + 6 * x ** 3 * y + 12 * x ** 2 * y ** 2 + 10 * x * y ** 3 + 3 * y ** 4
    h = x ** 5 + x ** 4 * y - 18 * x ** 3 * y ** 2 - 50 * x ** 2 * y ** 3 - 47 * x * y ** 4 - 15 * y ** 5
    assert groebner_lcm(f, g) == h