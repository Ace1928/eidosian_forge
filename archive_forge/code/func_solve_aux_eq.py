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
def solve_aux_eq(numa, dena, numy, deny, x, m):
    """
    Helper function to find a polynomial solution
    of degree m for the auxiliary differential
    equation.
    """
    psyms = symbols(f'C0:{m}', cls=Dummy)
    K = ZZ[psyms]
    psol = Poly(K.gens, x, domain=K) + Poly(x ** m, x, domain=K)
    auxeq = (dena * (numy.diff(x) * deny - numy * deny.diff(x) + numy ** 2) - numa * deny ** 2) * psol
    if m >= 1:
        px = psol.diff(x)
        auxeq += px * (2 * numy * deny * dena)
    if m >= 2:
        auxeq += px.diff(x) * (deny ** 2 * dena)
    if m != 0:
        return (psol, linsolve_dict(auxeq.all_coeffs(), psyms), True)
    else:
        return (S.One, auxeq, auxeq == 0)