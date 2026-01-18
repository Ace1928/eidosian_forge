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
def subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x], from
    the modified subresultant prs of p and q.

    The coefficients of the polynomials in these two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing 9(2) (2015), 123-138.

    """
    lst = modified_subresultants_pg(p, q, x)
    if lst == [] or len(lst) == 2:
        return lst
    lcf = LC(lst[0]) ** (degree(lst[0], x) - degree(lst[1], x))
    subr_seq = [lst[0], lst[1]]
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m - 1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]
    fact = [(-1) ** (j * (j - 1) / S(2)) for j in j_seq]
    lst_s = lst[2:]
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append(-lst_s[k] / lcf)
        else:
            subr_seq.append(lst_s[k] / lcf)
    return subr_seq