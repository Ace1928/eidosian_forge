from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Integer,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.abc import x, y, z
from sympy.external.gmpy import HAS_GMPY
from sympy.polys.domains import (ZZ, QQ, RR, CC, FF, GF, EX, EXRAW, ZZ_gmpy,
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.gaussiandomains import ZZ_I, QQ_I
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.realfield import RealField
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.rings import ring
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.fields import field
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.polyerrors import (
from sympy.testing.pytest import raises
from itertools import product
def test_PolynomialRing_from_FractionField():
    F, x, y = field('x,y', ZZ)
    R, X, Y = ring('x,y', ZZ)
    f = (x ** 2 + y ** 2) / (x + 1)
    g = (x ** 2 + y ** 2) / 4
    h = x ** 2 + y ** 2
    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X ** 2 / 4 + Y ** 2 / 4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X ** 2 + Y ** 2
    F, x, y = field('x,y', QQ)
    R, X, Y = ring('x,y', QQ)
    f = (x ** 2 + y ** 2) / (x + 1)
    g = (x ** 2 + y ** 2) / 4
    h = x ** 2 + y ** 2
    assert R.to_domain().from_FractionField(f, F.to_domain()) is None
    assert R.to_domain().from_FractionField(g, F.to_domain()) == X ** 2 / 4 + Y ** 2 / 4
    assert R.to_domain().from_FractionField(h, F.to_domain()) == X ** 2 + Y ** 2