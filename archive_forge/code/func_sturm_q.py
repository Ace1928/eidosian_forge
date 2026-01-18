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
def sturm_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Q[x].
    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).

    The coefficients of the polynomials in the Sturm sequence can be uniquely
    determined from the corresponding coefficients of the polynomials found
    either in:

        (a) the ``modified'' subresultant prs, (references 1, 2)

    or in

        (b) the subresultant prs (reference 3).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2 Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    if p == 0 or q == 0:
        return [p, q]
    d0 = degree(p, x)
    d1 = degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = (d1, d0)
        p, q = (q, p)
    if d0 > 0 and d1 == 0:
        return [p, q]
    flag = 0
    if LC(p, x) < 0:
        flag = 1
        p = -p
        q = -q
    a0, a1 = (p, q)
    sturm_seq = [a0, a1]
    a2 = -rem(a0, a1, domain=QQ)
    d2 = degree(a2, x)
    sturm_seq.append(a2)
    while d2 > 0:
        a0, a1, d0, d1 = (a1, a2, d1, d2)
        a2 = -rem(a0, a1, domain=QQ)
        d2 = degree(a2, x)
        sturm_seq.append(a2)
    if flag:
        sturm_seq = [-i for i in sturm_seq]
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)
    return sturm_seq