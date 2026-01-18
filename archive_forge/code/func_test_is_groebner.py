from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_is_groebner():
    R, x, y = ring('x,y', QQ, grlex)
    valid_groebner = [x ** 2, x * y, -QQ(1, 2) * x + y ** 2]
    invalid_groebner = [x ** 3, x * y, -QQ(1, 2) * x + y ** 2]
    assert is_groebner(valid_groebner, R) is True
    assert is_groebner(invalid_groebner, R) is False