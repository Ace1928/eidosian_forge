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
def limit_at_inf(num, den, x):
    """
    Find the limit of a rational function
    at oo
    """
    pwr = -val_at_inf(num, den, x)
    if pwr > 0:
        return oo * sign(num.LC() / den.LC())
    elif pwr == 0:
        return num.LC() / den.LC()
    else:
        return 0