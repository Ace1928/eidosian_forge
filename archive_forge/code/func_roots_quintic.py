import math
from functools import reduce
from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.numbers import Rational, igcd, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent
def roots_quintic(f):
    """
    Calculate exact roots of a solvable irreducible quintic with rational coefficients.
    Return an empty list if the quintic is reducible or not solvable.
    """
    result = []
    coeff_5, coeff_4, p_, q_, r_, s_ = f.all_coeffs()
    if not all((coeff.is_Rational for coeff in (coeff_5, coeff_4, p_, q_, r_, s_))):
        return result
    if coeff_5 != 1:
        f = Poly(f / coeff_5)
        _, coeff_4, p_, q_, r_, s_ = f.all_coeffs()
    if coeff_4:
        p = p_ - 2 * coeff_4 * coeff_4 / 5
        q = q_ - 3 * coeff_4 * p_ / 5 + 4 * coeff_4 ** 3 / 25
        r = r_ - 2 * coeff_4 * q_ / 5 + 3 * coeff_4 ** 2 * p_ / 25 - 3 * coeff_4 ** 4 / 125
        s = s_ - coeff_4 * r_ / 5 + coeff_4 ** 2 * q_ / 25 - coeff_4 ** 3 * p_ / 125 + 4 * coeff_4 ** 5 / 3125
        x = f.gen
        f = Poly(x ** 5 + p * x ** 3 + q * x ** 2 + r * x + s)
    else:
        p, q, r, s = (p_, q_, r_, s_)
    quintic = PolyQuintic(f)
    if not f.is_irreducible:
        return result
    f20 = quintic.f20
    if f20.is_irreducible:
        return result
    for _factor in f20.factor_list()[1]:
        if _factor[0].is_linear:
            theta = _factor[0].root(0)
            break
    d = discriminant(f)
    delta = sqrt(d)
    zeta1, zeta2, zeta3, zeta4 = quintic.zeta
    T = quintic.T(theta, d)
    tol = S(1e-10)
    alpha = T[1] + T[2] * delta
    alpha_bar = T[1] - T[2] * delta
    beta = T[3] + T[4] * delta
    beta_bar = T[3] - T[4] * delta
    disc = alpha ** 2 - 4 * beta
    disc_bar = alpha_bar ** 2 - 4 * beta_bar
    l0 = quintic.l0(theta)
    Stwo = S(2)
    l1 = _quintic_simplify((-alpha + sqrt(disc)) / Stwo)
    l4 = _quintic_simplify((-alpha - sqrt(disc)) / Stwo)
    l2 = _quintic_simplify((-alpha_bar + sqrt(disc_bar)) / Stwo)
    l3 = _quintic_simplify((-alpha_bar - sqrt(disc_bar)) / Stwo)
    order = quintic.order(theta, d)
    test = order * delta.n() - (l1.n() - l4.n()) * (l2.n() - l3.n())
    if not comp(test, 0, tol):
        l2, l3 = (l3, l2)
    R1 = l0 + l1 * zeta1 + l2 * zeta2 + l3 * zeta3 + l4 * zeta4
    R2 = l0 + l3 * zeta1 + l1 * zeta2 + l4 * zeta3 + l2 * zeta4
    R3 = l0 + l2 * zeta1 + l4 * zeta2 + l1 * zeta3 + l3 * zeta4
    R4 = l0 + l4 * zeta1 + l3 * zeta2 + l2 * zeta3 + l1 * zeta4
    Res = [None, [None] * 5, [None] * 5, [None] * 5, [None] * 5]
    Res_n = [None, [None] * 5, [None] * 5, [None] * 5, [None] * 5]
    R1 = _quintic_simplify(R1)
    R2 = _quintic_simplify(R2)
    R3 = _quintic_simplify(R3)
    R4 = _quintic_simplify(R4)
    x0 = z ** (S(1) / 5)
    x1 = sqrt(2)
    x2 = sqrt(5)
    x3 = sqrt(5 - x2)
    x4 = I * x2
    x5 = x4 + I
    x6 = I * x0 / 4
    x7 = x1 * sqrt(x2 + 5)
    sol = [x0, -x6 * (x1 * x3 - x5), x6 * (x1 * x3 + x5), -x6 * (x4 + x7 - I), x6 * (-x4 + x7 + I)]
    R1 = R1.as_real_imag()
    R2 = R2.as_real_imag()
    R3 = R3.as_real_imag()
    R4 = R4.as_real_imag()
    for i, s in enumerate(sol):
        Res[1][i] = _quintic_simplify(s.xreplace({z: R1[0] + I * R1[1]}))
        Res[2][i] = _quintic_simplify(s.xreplace({z: R2[0] + I * R2[1]}))
        Res[3][i] = _quintic_simplify(s.xreplace({z: R3[0] + I * R3[1]}))
        Res[4][i] = _quintic_simplify(s.xreplace({z: R4[0] + I * R4[1]}))
    for i in range(1, 5):
        for j in range(5):
            Res_n[i][j] = Res[i][j].n()
            Res[i][j] = _quintic_simplify(Res[i][j])
    r1 = Res[1][0]
    r1_n = Res_n[1][0]
    for i in range(5):
        if comp(im(r1_n * Res_n[4][i]), 0, tol):
            r4 = Res[4][i]
            break
    u, v = quintic.uv(theta, d)
    testplus = (u + v * delta * sqrt(5)).n()
    testminus = (u - v * delta * sqrt(5)).n()
    r4_n = r4.n()
    r2 = r3 = None
    for i in range(5):
        r2temp_n = Res_n[2][i]
        for j in range(5):
            r3temp_n = Res_n[3][j]
            if comp((r1_n * r2temp_n ** 2 + r4_n * r3temp_n ** 2 - testplus).n(), 0, tol) and comp((r3temp_n * r1_n ** 2 + r2temp_n * r4_n ** 2 - testminus).n(), 0, tol):
                r2 = Res[2][i]
                r3 = Res[3][j]
                break
        if r2 is not None:
            break
    else:
        return []
    x1 = (r1 + r2 + r3 + r4) / 5
    x2 = (r1 * zeta4 + r2 * zeta3 + r3 * zeta2 + r4 * zeta1) / 5
    x3 = (r1 * zeta3 + r2 * zeta1 + r3 * zeta4 + r4 * zeta2) / 5
    x4 = (r1 * zeta2 + r2 * zeta4 + r3 * zeta1 + r4 * zeta3) / 5
    x5 = (r1 * zeta1 + r2 * zeta2 + r3 * zeta3 + r4 * zeta4) / 5
    result = [x1, x2, x3, x4, x5]
    saw = set()
    for r in result:
        r = r.n(2)
        if r in saw:
            return []
        saw.add(r)
    if coeff_4:
        result = [x - coeff_4 / 5 for x in result]
    return result