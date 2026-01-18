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
def test_Domain_get_field():
    assert EX.has_assoc_Field is True
    assert ZZ.has_assoc_Field is True
    assert QQ.has_assoc_Field is True
    assert RR.has_assoc_Field is True
    assert ALG.has_assoc_Field is True
    assert ZZ[x].has_assoc_Field is True
    assert QQ[x].has_assoc_Field is True
    assert ZZ[x, y].has_assoc_Field is True
    assert QQ[x, y].has_assoc_Field is True
    assert EX.get_field() == EX
    assert ZZ.get_field() == QQ
    assert QQ.get_field() == QQ
    assert RR.get_field() == RR
    assert ALG.get_field() == ALG
    assert ZZ[x].get_field() == ZZ.frac_field(x)
    assert QQ[x].get_field() == QQ.frac_field(x)
    assert ZZ[x, y].get_field() == ZZ.frac_field(x, y)
    assert QQ[x, y].get_field() == QQ.frac_field(x, y)