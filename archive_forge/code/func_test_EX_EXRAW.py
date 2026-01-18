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
def test_EX_EXRAW():
    assert EXRAW.zero is S.Zero
    assert EXRAW.one is S.One
    assert EX(1) == EX.Expression(1)
    assert EX(1).ex is S.One
    assert EXRAW(1) is S.One
    assert 2 * EX((x + y * x) / x) == EX(2 + 2 * y) != 2 * ((x + y * x) / x)
    assert 2 * EXRAW((x + y * x) / x) == 2 * ((x + y * x) / x) != 1 + y
    assert EXRAW.convert_from(EX(1), EX) is EXRAW.one
    assert EX.convert_from(EXRAW(1), EXRAW) == EX.one
    assert EXRAW.from_sympy(S.One) is S.One
    assert EXRAW.to_sympy(EXRAW.one) is S.One
    raises(CoercionFailed, lambda: EXRAW.from_sympy([]))
    assert EXRAW.get_field() == EXRAW
    assert EXRAW.unify(EX) == EXRAW
    assert EX.unify(EXRAW) == EXRAW