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
def test_gaussian_domains():
    I = S.ImaginaryUnit
    a, b, c, d = [ZZ_I.convert(x) for x in (5, 2 + I, 3 - I, 5 - 5 * I)]
    assert ZZ_I.gcd(a, b) == b
    assert ZZ_I.gcd(a, c) == b
    assert ZZ_I.lcm(a, b) == a
    assert ZZ_I.lcm(a, c) == d
    assert ZZ_I(3, 4) != QQ_I(3, 4)
    assert ZZ_I(3, 0) != 3
    assert QQ_I(S(3) / 4, 0) != S(3) / 4
    assert ZZ_I(0, 0).quadrant() == 0
    assert ZZ_I(-1, 0).quadrant() == 2
    assert QQ_I.convert(QQ(3, 2)) == QQ_I(QQ(3, 2), QQ(0))
    assert QQ_I.convert(QQ(3, 2), QQ) == QQ_I(QQ(3, 2), QQ(0))
    for G in (QQ_I, ZZ_I):
        q = G(3, 4)
        assert str(q) == '3 + 4*I'
        assert q.parent() == G
        assert q._get_xy(pi) == (None, None)
        assert q._get_xy(2) == (2, 0)
        assert q._get_xy(2 * I) == (0, 2)
        assert hash(q) == hash((3, 4))
        assert G(1, 2) == G(1, 2)
        assert G(1, 2) != G(1, 3)
        assert G(3, 0) == G(3)
        assert q + q == G(6, 8)
        assert q - q == G(0, 0)
        assert 3 - q == -q + 3 == G(0, -4)
        assert 3 + q == q + 3 == G(6, 4)
        assert q * q == G(-7, 24)
        assert 3 * q == q * 3 == G(9, 12)
        assert q ** 0 == G(1, 0)
        assert q ** 1 == q
        assert q ** 2 == q * q == G(-7, 24)
        assert q ** 3 == q * q * q == G(-117, 44)
        assert 1 / q == q ** (-1) == QQ_I(S(3) / 25, -S(4) / 25)
        assert q / 1 == QQ_I(3, 4)
        assert q / 2 == QQ_I(S(3) / 2, 2)
        assert q / 3 == QQ_I(1, S(4) / 3)
        assert 3 / q == QQ_I(S(9) / 25, -S(12) / 25)
        i, r = divmod(q, 2)
        assert 2 * i + r == q
        i, r = divmod(2, q)
        assert q * i + r == G(2, 0)
        raises(ZeroDivisionError, lambda: q % 0)
        raises(ZeroDivisionError, lambda: q / 0)
        raises(ZeroDivisionError, lambda: q // 0)
        raises(ZeroDivisionError, lambda: divmod(q, 0))
        raises(ZeroDivisionError, lambda: divmod(q, 0))
        raises(TypeError, lambda: q + x)
        raises(TypeError, lambda: q - x)
        raises(TypeError, lambda: x + q)
        raises(TypeError, lambda: x - q)
        raises(TypeError, lambda: q * x)
        raises(TypeError, lambda: x * q)
        raises(TypeError, lambda: q / x)
        raises(TypeError, lambda: x / q)
        raises(TypeError, lambda: q // x)
        raises(TypeError, lambda: x // q)
        assert G.from_sympy(S(2)) == G(2, 0)
        assert G.to_sympy(G(2, 0)) == S(2)
        raises(CoercionFailed, lambda: G.from_sympy(pi))
        PR = G.inject(x)
        assert isinstance(PR, PolynomialRing)
        assert PR.domain == G
        assert len(PR.gens) == 1 and PR.gens[0].as_expr() == x
        if G is QQ_I:
            AF = G.as_AlgebraicField()
            assert isinstance(AF, AlgebraicField)
            assert AF.domain == QQ
            assert AF.ext.args[0] == I
        for qi in [G(-1, 0), G(1, 0), G(0, -1), G(0, 1)]:
            assert G.is_negative(qi) is False
            assert G.is_positive(qi) is False
            assert G.is_nonnegative(qi) is False
            assert G.is_nonpositive(qi) is False
        domains = [ZZ_python(), QQ_python(), AlgebraicField(QQ, I)]
        if HAS_GMPY:
            domains += [ZZ_gmpy(), QQ_gmpy()]
        for K in domains:
            assert G.convert(K(2)) == G(2, 0)
            assert G.convert(K(2), K) == G(2, 0)
        for K in (ZZ_I, QQ_I):
            assert G.convert(K(1, 1)) == G(1, 1)
            assert G.convert(K(1, 1), K) == G(1, 1)
        if G == ZZ_I:
            assert repr(q) == 'ZZ_I(3, 4)'
            assert q // 3 == G(1, 1)
            assert 12 // q == G(1, -2)
            assert 12 % q == G(1, 2)
            assert q % 2 == G(-1, 0)
            assert i == G(0, 0)
            assert r == G(2, 0)
            assert G.get_ring() == G
            assert G.get_field() == QQ_I
        else:
            assert repr(q) == 'QQ_I(3, 4)'
            assert G.get_ring() == ZZ_I
            assert G.get_field() == G
            assert q // 3 == G(1, S(4) / 3)
            assert 12 // q == G(S(36) / 25, -S(48) / 25)
            assert 12 % q == G(0, 0)
            assert q % 2 == G(0, 0)
            assert i == G(S(6) / 25, -S(8) / 25), (G, i)
            assert r == G(0, 0)
            q2 = G(S(3) / 2, S(5) / 3)
            assert G.numer(q2) == ZZ_I(9, 10)
            assert G.denom(q2) == ZZ_I(6)