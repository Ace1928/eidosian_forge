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
def test_Domain_unify_algebraic():
    sqrt5 = QQ.algebraic_field(sqrt(5))
    sqrt7 = QQ.algebraic_field(sqrt(7))
    sqrt57 = QQ.algebraic_field(sqrt(5), sqrt(7))
    assert sqrt5.unify(sqrt7) == sqrt57
    assert sqrt5.unify(sqrt5[x, y]) == sqrt5[x, y]
    assert sqrt5[x, y].unify(sqrt5) == sqrt5[x, y]
    assert sqrt5.unify(sqrt5.frac_field(x, y)) == sqrt5.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt5) == sqrt5.frac_field(x, y)
    assert sqrt5.unify(sqrt7[x, y]) == sqrt57[x, y]
    assert sqrt5[x, y].unify(sqrt7) == sqrt57[x, y]
    assert sqrt5.unify(sqrt7.frac_field(x, y)) == sqrt57.frac_field(x, y)
    assert sqrt5.frac_field(x, y).unify(sqrt7) == sqrt57.frac_field(x, y)