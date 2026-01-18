from itertools import product
from sympy.core import S
from sympy.core.add import Add
from sympy.core.numbers import oo, Float
from sympy.core.function import count_ops
from sympy.core.relational import Eq
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions import sqrt, exp
from sympy.functions.elementary.complexes import sign
from sympy.integrals.integrals import Integral
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyroots import roots
from sympy.solvers.solveset import linsolve
def compute_m_ybar(x, poles, choice, N):
    """
    Helper function to calculate -

    1. m - The degree bound for the polynomial
    solution that must be found for the auxiliary
    differential equation.

    2. ybar - Part of the solution which can be
    computed using the poles, c and d vectors.
    """
    ybar = 0
    m = Poly(choice[-1][-1], x, extension=True)
    dybar = []
    for i, polei in enumerate(poles):
        for j, cij in enumerate(choice[i]):
            dybar.append(cij / (x - polei) ** (j + 1))
        m -= Poly(choice[i][0], x, extension=True)
    ybar += Add(*dybar)
    for i in range(N + 1):
        ybar += choice[-1][i] * x ** i
    return (m.expr, ybar)