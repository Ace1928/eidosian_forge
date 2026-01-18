from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError
def subresultants_amv(f, g, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(p, q, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Akritas-Malaschonok-Vigklas Theorem of 2015 and the Collins-Brown-
    Traub formula for coefficient reduction.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    if f == 0 or g == 0:
        return [f, g]
    d0 = degree(f, x)
    d1 = degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = (d1, d0)
        f, g = (g, f)
    if d0 > 0 and d1 == 0:
        return [f, g]
    a0 = f
    a1 = g
    subres_l = [a0, a1]
    deg_dif_p1, c = (degree(a0, x) - degree(a1, x) + 1, -1)
    sigma1 = LC(a1, x)
    i, s = (0, 0)
    p_odd_index_sum = 0
    p0 = deg_dif_p1 - 1
    if p0 % 2 == 1:
        s += 1
    phi = floor((s + 1) / 2)
    i += 1
    a2 = rem_z(a0, a1, x) / Abs((-1) ** deg_dif_p1)
    sigma2 = LC(a2, x)
    d2 = degree(a2, x)
    p1 = d1 - d2
    sgn_den = compute_sign(sigma1, p0 + 1)
    psi = i + phi + p_odd_index_sum
    num = (-1) ** psi
    den = sgn_den
    if sign(num / den) > 0:
        subres_l.append(a2)
    else:
        subres_l.append(-a2)
    if p1 % 2 == 1:
        s += 1
    if p1 - 1 > 0:
        sgn_den = sgn_den * compute_sign(sigma1, p1 - 1)
    while d2 >= 1:
        phi = floor((s + 1) / 2)
        if i % 2 == 1:
            p_odd_index_sum += p1
        a0, a1, d0, d1 = (a1, a2, d1, d2)
        p0 = p1
        i += 1
        sigma0 = -LC(a0)
        c = sigma0 ** (deg_dif_p1 - 1) / c ** (deg_dif_p1 - 2)
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs(c ** (deg_dif_p1 - 1) * sigma0)
        sigma3 = LC(a2, x)
        d2 = degree(a2, x)
        p1 = d1 - d2
        psi = i + phi + p_odd_index_sum
        sigma1, sigma2 = (sigma2, sigma3)
        sgn_den = compute_sign(sigma1, p0 + 1) * sgn_den
        num = (-1) ** psi
        den = sgn_den
        if sign(num / den) > 0:
            subres_l.append(a2)
        else:
            subres_l.append(-a2)
        if p1 % 2 == 1:
            s += 1
        if p1 - 1 > 0:
            sgn_den = sgn_den * compute_sign(sigma1, p1 - 1)
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)
    return subres_l