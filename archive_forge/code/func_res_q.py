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
def res_q(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Q[x], using the function rem.
    See Cohen's book p. 281.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1) ** (m * n) * res_q(g, f, x)
    elif n == 0:
        return g ** m
    else:
        r = rem(f, g, x)
        if r == 0:
            return 0
        else:
            s = degree(r, x)
            l = LC(g, x)
            return (-1) ** (m * n) * l ** (m - s) * res_q(g, r, x)