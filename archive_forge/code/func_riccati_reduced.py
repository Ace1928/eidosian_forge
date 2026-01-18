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
def riccati_reduced(eq, f, x):
    """
    Convert a Riccati ODE into its corresponding
    normal Riccati ODE.
    """
    match, funcs = match_riccati(eq, f, x)
    if not match:
        return False
    b0, b1, b2 = funcs
    a = -b0 * b2 + b1 ** 2 / 4 - b1.diff(x) / 2 + 3 * b2.diff(x) ** 2 / (4 * b2 ** 2) + b1 * b2.diff(x) / (2 * b2) - b2.diff(x, 2) / (2 * b2)
    return f(x).diff(x) + f(x) ** 2 - a