from sympy.core.numbers import (AlgebraicNumber, I, pi, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.external.gmpy import MPQ
from sympy.polys.numberfields.subfield import (
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises
from sympy.abc import x
def test_field_isomorphism():
    assert field_isomorphism(3, sqrt(2)) == [3]
    assert field_isomorphism(I * sqrt(3), I * sqrt(3) / 2) == [2, 0]
    assert field_isomorphism(-I * sqrt(3), I * sqrt(3) / 2) == [-2, 0]
    assert field_isomorphism(I * sqrt(3), -I * sqrt(3) / 2) == [-2, 0]
    assert field_isomorphism(-I * sqrt(3), -I * sqrt(3) / 2) == [2, 0]
    assert field_isomorphism(2 * I * sqrt(3) / 7, 5 * I * sqrt(3) / 3) == [Rational(6, 35), 0]
    assert field_isomorphism(-2 * I * sqrt(3) / 7, 5 * I * sqrt(3) / 3) == [Rational(-6, 35), 0]
    assert field_isomorphism(2 * I * sqrt(3) / 7, -5 * I * sqrt(3) / 3) == [Rational(-6, 35), 0]
    assert field_isomorphism(-2 * I * sqrt(3) / 7, -5 * I * sqrt(3) / 3) == [Rational(6, 35), 0]
    assert field_isomorphism(2 * I * sqrt(3) / 7 + 27, 5 * I * sqrt(3) / 3) == [Rational(6, 35), 27]
    assert field_isomorphism(-2 * I * sqrt(3) / 7 + 27, 5 * I * sqrt(3) / 3) == [Rational(-6, 35), 27]
    assert field_isomorphism(2 * I * sqrt(3) / 7 + 27, -5 * I * sqrt(3) / 3) == [Rational(-6, 35), 27]
    assert field_isomorphism(-2 * I * sqrt(3) / 7 + 27, -5 * I * sqrt(3) / 3) == [Rational(6, 35), 27]
    p = AlgebraicNumber(sqrt(2) + sqrt(3))
    q = AlgebraicNumber(-sqrt(2) + sqrt(3))
    r = AlgebraicNumber(sqrt(2) - sqrt(3))
    s = AlgebraicNumber(-sqrt(2) - sqrt(3))
    pos_coeffs = [S.Half, S.Zero, Rational(-9, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(9, 2), S.Zero]
    a = AlgebraicNumber(sqrt(2))
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True
    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs
    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs
    a = AlgebraicNumber(-sqrt(2))
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True
    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs
    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs
    pos_coeffs = [S.Half, S.Zero, Rational(-11, 2), S.Zero]
    neg_coeffs = [Rational(-1, 2), S.Zero, Rational(11, 2), S.Zero]
    a = AlgebraicNumber(sqrt(3))
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True
    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs
    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs
    a = AlgebraicNumber(-sqrt(3))
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True
    assert field_isomorphism(a, p, fast=True) == pos_coeffs
    assert field_isomorphism(a, q, fast=True) == pos_coeffs
    assert field_isomorphism(a, r, fast=True) == neg_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_coeffs
    assert field_isomorphism(a, p, fast=False) == pos_coeffs
    assert field_isomorphism(a, q, fast=False) == pos_coeffs
    assert field_isomorphism(a, r, fast=False) == neg_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_coeffs
    pos_coeffs = [Rational(3, 2), S.Zero, Rational(-33, 2), -S(8)]
    neg_coeffs = [Rational(-3, 2), S.Zero, Rational(33, 2), -S(8)]
    a = AlgebraicNumber(3 * sqrt(3) - 8)
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True
    assert field_isomorphism(a, p, fast=True) == neg_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_coeffs
    assert field_isomorphism(a, s, fast=True) == pos_coeffs
    assert field_isomorphism(a, p, fast=False) == neg_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_coeffs
    assert field_isomorphism(a, s, fast=False) == pos_coeffs
    a = AlgebraicNumber(3 * sqrt(2) + 2 * sqrt(3) + 1)
    pos_1_coeffs = [S.Half, S.Zero, Rational(-5, 2), S.One]
    neg_5_coeffs = [Rational(-5, 2), S.Zero, Rational(49, 2), S.One]
    pos_5_coeffs = [Rational(5, 2), S.Zero, Rational(-49, 2), S.One]
    neg_1_coeffs = [Rational(-1, 2), S.Zero, Rational(5, 2), S.One]
    assert is_isomorphism_possible(a, p) is True
    assert is_isomorphism_possible(a, q) is True
    assert is_isomorphism_possible(a, r) is True
    assert is_isomorphism_possible(a, s) is True
    assert field_isomorphism(a, p, fast=True) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=True) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=True) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=True) == neg_1_coeffs
    assert field_isomorphism(a, p, fast=False) == pos_1_coeffs
    assert field_isomorphism(a, q, fast=False) == neg_5_coeffs
    assert field_isomorphism(a, r, fast=False) == pos_5_coeffs
    assert field_isomorphism(a, s, fast=False) == neg_1_coeffs
    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(sqrt(3))
    c = AlgebraicNumber(sqrt(7))
    assert is_isomorphism_possible(a, b) is True
    assert is_isomorphism_possible(b, a) is True
    assert is_isomorphism_possible(c, p) is False
    assert field_isomorphism(sqrt(2), sqrt(3), fast=True) is None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=True) is None
    assert field_isomorphism(sqrt(2), sqrt(3), fast=False) is None
    assert field_isomorphism(sqrt(3), sqrt(2), fast=False) is None
    a = AlgebraicNumber(sqrt(2))
    b = AlgebraicNumber(2 ** (S(1) / 3))
    assert is_isomorphism_possible(a, b) is False
    assert field_isomorphism(a, b) is None