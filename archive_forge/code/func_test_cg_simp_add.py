from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.cg import Wigner3j, Wigner6j, Wigner9j, CG, cg_simp
from sympy.functions.special.tensor_functions import KroneckerDelta
def test_cg_simp_add():
    j, m1, m1p, m2, m2p = symbols('j m1 m1p m2 m2p')
    a = CG(S.Half, S.Half, 0, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), 0, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, 0, 0, 1, 1)
    d = CG(1, 0, 0, 0, 1, 0)
    e = CG(1, -1, 0, 0, 1, -1)
    assert cg_simp(a + b) == 2
    assert cg_simp(c + d + e) == 3
    assert cg_simp(a + b + c + d + e) == 5
    assert cg_simp(a + b + c) == 2 + c
    assert cg_simp(2 * a + b) == 2 + a
    assert cg_simp(2 * c + d + e) == 3 + c
    assert cg_simp(5 * a + 5 * b) == 10
    assert cg_simp(5 * c + 5 * d + 5 * e) == 15
    assert cg_simp(-a - b) == -2
    assert cg_simp(-c - d - e) == -3
    assert cg_simp(-6 * a - 6 * b) == -12
    assert cg_simp(-4 * c - 4 * d - 4 * e) == -12
    a = CG(S.Half, S.Half, j, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), j, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, j, 0, 1, 1)
    d = CG(1, 0, j, 0, 1, 0)
    e = CG(1, -1, j, 0, 1, -1)
    assert cg_simp(a + b) == 2 * KroneckerDelta(j, 0)
    assert cg_simp(c + d + e) == 3 * KroneckerDelta(j, 0)
    assert cg_simp(a + b + c + d + e) == 5 * KroneckerDelta(j, 0)
    assert cg_simp(a + b + c) == 2 * KroneckerDelta(j, 0) + c
    assert cg_simp(2 * a + b) == 2 * KroneckerDelta(j, 0) + a
    assert cg_simp(2 * c + d + e) == 3 * KroneckerDelta(j, 0) + c
    assert cg_simp(5 * a + 5 * b) == 10 * KroneckerDelta(j, 0)
    assert cg_simp(5 * c + 5 * d + 5 * e) == 15 * KroneckerDelta(j, 0)
    assert cg_simp(-a - b) == -2 * KroneckerDelta(j, 0)
    assert cg_simp(-c - d - e) == -3 * KroneckerDelta(j, 0)
    assert cg_simp(-6 * a - 6 * b) == -12 * KroneckerDelta(j, 0)
    assert cg_simp(-4 * c - 4 * d - 4 * e) == -12 * KroneckerDelta(j, 0)
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, -1, 0, 0)
    d = CG(1, 0, 1, 0, 0, 0)
    e = CG(1, -1, 1, 1, 0, 0)
    assert cg_simp(a - b) == sqrt(2)
    assert cg_simp(c - d + e) == sqrt(3)
    assert cg_simp(a - b + c - d + e) == sqrt(2) + sqrt(3)
    assert cg_simp(a - b + c) == sqrt(2) + c
    assert cg_simp(2 * a - b) == sqrt(2) + a
    assert cg_simp(2 * c - d + e) == sqrt(3) + c
    assert cg_simp(5 * a - 5 * b) == 5 * sqrt(2)
    assert cg_simp(5 * c - 5 * d + 5 * e) == 5 * sqrt(3)
    assert cg_simp(-a + b) == -sqrt(2)
    assert cg_simp(-c + d - e) == -sqrt(3)
    assert cg_simp(-6 * a + 6 * b) == -6 * sqrt(2)
    assert cg_simp(-4 * c + 4 * d - 4 * e) == -4 * sqrt(3)
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), j, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, j, 0)
    c = CG(1, 1, 1, -1, j, 0)
    d = CG(1, 0, 1, 0, j, 0)
    e = CG(1, -1, 1, 1, j, 0)
    assert cg_simp(a - b) == sqrt(2) * KroneckerDelta(j, 0)
    assert cg_simp(c - d + e) == sqrt(3) * KroneckerDelta(j, 0)
    assert cg_simp(a - b + c - d + e) == sqrt(2) * KroneckerDelta(j, 0) + sqrt(3) * KroneckerDelta(j, 0)
    assert cg_simp(a - b + c) == sqrt(2) * KroneckerDelta(j, 0) + c
    assert cg_simp(2 * a - b) == sqrt(2) * KroneckerDelta(j, 0) + a
    assert cg_simp(2 * c - d + e) == sqrt(3) * KroneckerDelta(j, 0) + c
    assert cg_simp(5 * a - 5 * b) == 5 * sqrt(2) * KroneckerDelta(j, 0)
    assert cg_simp(5 * c - 5 * d + 5 * e) == 5 * sqrt(3) * KroneckerDelta(j, 0)
    assert cg_simp(-a + b) == -sqrt(2) * KroneckerDelta(j, 0)
    assert cg_simp(-c + d - e) == -sqrt(3) * KroneckerDelta(j, 0)
    assert cg_simp(-6 * a + 6 * b) == -6 * sqrt(2) * KroneckerDelta(j, 0)
    assert cg_simp(-4 * c + 4 * d - 4 * e) == -4 * sqrt(3) * KroneckerDelta(j, 0)
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0) ** 2
    b = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0) ** 2
    c = CG(1, 0, 1, 1, 1, 1) ** 2
    d = CG(1, 0, 1, 1, 2, 1) ** 2
    assert cg_simp(a + b) == 1
    assert cg_simp(c + d) == 1
    assert cg_simp(a + b + c + d) == 2
    assert cg_simp(4 * a + 4 * b) == 4
    assert cg_simp(4 * c + 4 * d) == 4
    assert cg_simp(5 * a + 3 * b) == 3 + 2 * a
    assert cg_simp(5 * c + 3 * d) == 3 + 2 * c
    assert cg_simp(-a - b) == -1
    assert cg_simp(-c - d) == -1
    a = CG(S.Half, m1, S.Half, m2, 1, 1) ** 2
    b = CG(S.Half, m1, S.Half, m2, 1, 0) ** 2
    c = CG(S.Half, m1, S.Half, m2, 1, -1) ** 2
    d = CG(S.Half, m1, S.Half, m2, 0, 0) ** 2
    assert cg_simp(a + b + c + d) == 1
    assert cg_simp(4 * a + 4 * b + 4 * c + 4 * d) == 4
    assert cg_simp(3 * a + 5 * b + 3 * c + 4 * d) == 3 + 2 * b + d
    assert cg_simp(-a - b - c - d) == -1
    a = CG(1, m1, 1, m2, 2, 2) ** 2
    b = CG(1, m1, 1, m2, 2, 1) ** 2
    c = CG(1, m1, 1, m2, 2, 0) ** 2
    d = CG(1, m1, 1, m2, 2, -1) ** 2
    e = CG(1, m1, 1, m2, 2, -2) ** 2
    f = CG(1, m1, 1, m2, 1, 1) ** 2
    g = CG(1, m1, 1, m2, 1, 0) ** 2
    h = CG(1, m1, 1, m2, 1, -1) ** 2
    i = CG(1, m1, 1, m2, 0, 0) ** 2
    assert cg_simp(a + b + c + d + e + f + g + h + i) == 1
    assert cg_simp(4 * (a + b + c + d + e + f + g + h + i)) == 4
    assert cg_simp(a + b + 2 * c + d + 4 * e + f + g + h + i) == 1 + c + 3 * e
    assert cg_simp(-a - b - c - d - e - f - g - h - i) == -1
    a = CG(S.Half, S(1) / 2, S.Half, Rational(-1, 2), 1, 0) * CG(S.Half, Rational(-1, 2), S.Half, S.Half, 1, 0)
    b = CG(S.Half, S(1) / 2, S.Half, Rational(-1, 2), 0, 0) * CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, 0, 2, 1) * CG(1, 0, 1, 1, 2, 1)
    d = CG(1, 1, 1, 0, 1, 1) * CG(1, 0, 1, 1, 1, 1)
    assert cg_simp(a + b) == 0
    assert cg_simp(c + d) == 0
    a = CG(S.Half, m1, S.Half, m2, 1, 1) * CG(S.Half, m1p, S.Half, m2p, 1, 1)
    b = CG(S.Half, m1, S.Half, m2, 1, 0) * CG(S.Half, m1p, S.Half, m2p, 1, 0)
    c = CG(S.Half, m1, S.Half, m2, 1, -1) * CG(S.Half, m1p, S.Half, m2p, 1, -1)
    d = CG(S.Half, m1, S.Half, m2, 0, 0) * CG(S.Half, m1p, S.Half, m2p, 0, 0)
    assert cg_simp(a + b + c + d) == KroneckerDelta(m1, m1p) * KroneckerDelta(m2, m2p)
    a = CG(1, m1, 1, m2, 2, 2) * CG(1, m1p, 1, m2p, 2, 2)
    b = CG(1, m1, 1, m2, 2, 1) * CG(1, m1p, 1, m2p, 2, 1)
    c = CG(1, m1, 1, m2, 2, 0) * CG(1, m1p, 1, m2p, 2, 0)
    d = CG(1, m1, 1, m2, 2, -1) * CG(1, m1p, 1, m2p, 2, -1)
    e = CG(1, m1, 1, m2, 2, -2) * CG(1, m1p, 1, m2p, 2, -2)
    f = CG(1, m1, 1, m2, 1, 1) * CG(1, m1p, 1, m2p, 1, 1)
    g = CG(1, m1, 1, m2, 1, 0) * CG(1, m1p, 1, m2p, 1, 0)
    h = CG(1, m1, 1, m2, 1, -1) * CG(1, m1p, 1, m2p, 1, -1)
    i = CG(1, m1, 1, m2, 0, 0) * CG(1, m1p, 1, m2p, 0, 0)
    assert cg_simp(a + b + c + d + e + f + g + h + i) == KroneckerDelta(m1, m1p) * KroneckerDelta(m2, m2p)