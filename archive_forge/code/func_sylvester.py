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
def sylvester(f, g, x, method=1):
    """
      The input polynomials f, g are in Z[x] or in Q[x]. Let m = degree(f, x),
      n = degree(g, x) and mx = max(m, n).

      a. If method = 1 (default), computes sylvester1, Sylvester's matrix of 1840
          of dimension (m + n) x (m + n). The determinants of properly chosen
          submatrices of this matrix (a.k.a. subresultants) can be
          used to compute the coefficients of the Euclidean PRS of f, g.

      b. If method = 2, computes sylvester2, Sylvester's matrix of 1853
          of dimension (2*mx) x (2*mx). The determinants of properly chosen
          submatrices of this matrix (a.k.a. ``modified'' subresultants) can be
          used to compute the coefficients of the Sturmian PRS of f, g.

      Applications of these Matrices can be found in the references below.
      Especially, for applications of sylvester2, see the first reference!!

      References
      ==========
      1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
      by Van Vleck Regarding Sturm Sequences. Serdica Journal of Computing,
      Vol. 7, No 4, 101-134, 2013.

      2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
      and Modified Subresultant Polynomial Remainder Sequences.''
      Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    m, n = (degree(Poly(f, x), x), degree(Poly(g, x), x))
    if m == n and n < 0:
        return Matrix([])
    if m == n and n == 0:
        return Matrix([])
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])
    fp = Poly(f, x).all_coeffs()
    gp = Poly(g, x).all_coeffs()
    if method <= 1:
        M = zeros(m + n)
        k = 0
        for i in range(n):
            j = k
            for coeff in fp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        k = 0
        for i in range(n, m + n):
            j = k
            for coeff in gp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        return M
    if method >= 2:
        if len(fp) < len(gp):
            h = []
            for i in range(len(gp) - len(fp)):
                h.append(0)
            fp[:0] = h
        else:
            h = []
            for i in range(len(fp) - len(gp)):
                h.append(0)
            gp[:0] = h
        mx = max(m, n)
        dim = 2 * mx
        M = zeros(dim)
        k = 0
        for i in range(mx):
            j = k
            for coeff in fp:
                M[2 * i, j] = coeff
                j = j + 1
            j = k
            for coeff in gp:
                M[2 * i + 1, j] = coeff
                j = j + 1
            k = k + 1
        return M