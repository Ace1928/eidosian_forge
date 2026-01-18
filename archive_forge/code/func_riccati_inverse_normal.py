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
def riccati_inverse_normal(y, x, b1, b2, bp=None):
    """
    Inverse transforming the solution to the normal
    Riccati ODE to get the solution to the Riccati ODE.
    """
    if bp is None:
        bp = -b2.diff(x) / (2 * b2 ** 2) - b1 / (2 * b2)
    return -y / b2 + bp