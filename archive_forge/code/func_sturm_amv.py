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
def sturm_amv(p, q, x, method=0):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].
    If q = diff(p, x, 1) it is the usual Sturm sequence.

    A. If method == 0, default, the remainder coefficients of the
       sequence are (in absolute value) ``modified'' subresultants, which
       for non-monic polynomials are greater than the coefficients of the
       corresponding subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    B. If method == 1, the remainder coefficients of the sequence are (in
       absolute value) subresultants, which for non-monic polynomials are
       smaller than the coefficients of the corresponding ``modified''
       subresultants by the factor Abs( LC(p)**( deg(p)- deg(q)) ).

    If the Sturm sequence is complete, method=0 and LC( p ) > 0, then the
    coefficients of the polynomials in the sequence are ``modified'' subresultants.
    That is, they are  determinants of appropriately selected submatrices of
    sylvester2, Sylvester's matrix of 1853. In this case the Sturm sequence
    coincides with the ``modified'' subresultant prs, of the polynomials
    p, q.

    If the Sturm sequence is incomplete and method=0 then the signs of the
    coefficients of the polynomials in the sequence may differ from the signs
    of the coefficients of the corresponding polynomials in the ``modified''
    subresultant prs; however, the absolute values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, we first compute the euclidean sequence  of p and q using
    euclid_amv(p, q, x) and then: (a) change the signs of the remainders in the
    Euclidean sequence according to the pattern "-, -, +, +, -, -, +, +,..."
    (see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference)
    and (b) if method=0, assuming deg(p) > deg(q), we multiply the remainder
    coefficients of the Euclidean sequence times the factor
    Abs( LC(p)**( deg(p)- deg(q)) ) to make them modified subresultants.
    See also the function sturm_pg(p, q, x).

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.'' Serdica
    Journal of Computing 9(2) (2015), 123-138.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    prs = euclid_amv(p, q, x)
    if prs == [] or len(prs) == 2:
        return prs
    lcf = Abs(LC(prs[0]) ** (degree(prs[0], x) - degree(prs[1], x)))
    sturm_seq = [prs[0], prs[1]]
    flag = 0
    m = len(prs)
    i = 2
    while i <= m - 1:
        if flag == 0:
            sturm_seq.append(-prs[i])
            i = i + 1
            if i == m:
                break
            sturm_seq.append(-prs[i])
            i = i + 1
            flag = 1
        elif flag == 1:
            sturm_seq.append(prs[i])
            i = i + 1
            if i == m:
                break
            sturm_seq.append(prs[i])
            i = i + 1
            flag = 0
    if method == 0 and lcf > 1:
        aux_seq = [sturm_seq[0], sturm_seq[1]]
        for i in range(2, m):
            aux_seq.append(simplify(sturm_seq[i] * lcf))
        sturm_seq = aux_seq
    return sturm_seq