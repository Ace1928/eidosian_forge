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
def test_RR_Float():
    f1 = Float('1.01')
    f2 = Float('1.0000000000000000000001')
    assert f1._prec == 53
    assert f2._prec == 80
    assert RR(f1) - 1 > 1e-50
    assert RR(f2) - 1 < 1e-50
    RR2 = RealField(prec=f2._prec)
    assert RR2(f1) - 1 > 1e-50
    assert RR2(f2) - 1 > 1e-50