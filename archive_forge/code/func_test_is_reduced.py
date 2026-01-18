from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_is_reduced():
    R, x, y = ring('x,y', QQ, lex)
    f = x ** 2 + 2 * x * y ** 2
    g = x * y + 2 * y ** 3 - 1
    assert is_reduced([f, g], R) == False
    G = groebner([f, g], R)
    assert is_reduced(G, R) == True