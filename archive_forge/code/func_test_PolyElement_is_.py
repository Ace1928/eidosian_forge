from functools import reduce
from operator import add, mul
from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import (oo, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
def test_PolyElement_is_():
    R, x, y, z = ring('x,y,z', QQ)
    assert (x - x).is_generator == False
    assert (x - x).is_ground == True
    assert (x - x).is_monomial == True
    assert (x - x).is_term == True
    assert (x - x + 1).is_generator == False
    assert (x - x + 1).is_ground == True
    assert (x - x + 1).is_monomial == True
    assert (x - x + 1).is_term == True
    assert x.is_generator == True
    assert x.is_ground == False
    assert x.is_monomial == True
    assert x.is_term == True
    assert (x * y).is_generator == False
    assert (x * y).is_ground == False
    assert (x * y).is_monomial == True
    assert (x * y).is_term == True
    assert (3 * x).is_generator == False
    assert (3 * x).is_ground == False
    assert (3 * x).is_monomial == False
    assert (3 * x).is_term == True
    assert (3 * x + 1).is_generator == False
    assert (3 * x + 1).is_ground == False
    assert (3 * x + 1).is_monomial == False
    assert (3 * x + 1).is_term == False
    assert R(0).is_zero is True
    assert R(1).is_zero is False
    assert R(0).is_one is False
    assert R(1).is_one is True
    assert (x - 1).is_monic is True
    assert (2 * x - 1).is_monic is False
    assert (3 * x + 2).is_primitive is True
    assert (4 * x + 2).is_primitive is False
    assert (x + y + z + 1).is_linear is True
    assert (x * y * z + 1).is_linear is False
    assert (x * y + z + 1).is_quadratic is True
    assert (x * y * z + 1).is_quadratic is False
    assert (x - 1).is_squarefree is True
    assert ((x - 1) ** 2).is_squarefree is False
    assert (x ** 2 + x + 1).is_irreducible is True
    assert (x ** 2 + 2 * x + 1).is_irreducible is False
    _, t = ring('t', FF(11))
    assert (7 * t + 3).is_irreducible is True
    assert (7 * t ** 2 + 3 * t + 1).is_irreducible is False
    _, u = ring('u', ZZ)
    f = u ** 16 + u ** 14 - u ** 10 - u ** 8 - u ** 6 + u ** 2
    assert f.is_cyclotomic is False
    assert (f + 1).is_cyclotomic is True
    raises(MultivariatePolynomialError, lambda: x.is_cyclotomic)
    R, = ring('', ZZ)
    assert R(4).is_squarefree is True
    assert R(6).is_irreducible is True