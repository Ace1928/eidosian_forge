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
def subresultants_sylv(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 1). The dimensions of the
    latter are (deg(f) + deg(g)) x (deg(f) + deg(g)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 1).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials f, g.

    References:
    ===========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    if f == 0 or g == 0:
        return [f, g]
    n = degF = degree(f, x)
    m = degG = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = (m, n, degG, degF, g, f)
    if n > 0 and m == 0:
        return [f, g]
    SR_L = [f, g]
    S = sylvester(f, g, x, 1)
    j = m - 1
    while j > 0:
        Sp = S[:, :]
        for ind in range(m + n - j, m + n):
            Sp.row_del(m + n - j)
        for ind in range(m - j, m):
            Sp.row_del(m - j)
        coeff_L, k, l = ([], Sp.rows, 0)
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1
    SR_L.append(S.det())
    return process_matrix_output(SR_L, x)