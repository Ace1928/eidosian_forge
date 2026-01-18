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
def construct_c(num, den, x, poles, muls):
    """
    Helper function to calculate the coefficients
    in the c-vector for each pole.
    """
    c = []
    for pole, mul in zip(poles, muls):
        c.append([])
        if mul == 1:
            c[-1].extend(construct_c_case_3())
        elif mul == 2:
            c[-1].extend(construct_c_case_1(num, den, x, pole))
        else:
            c[-1].extend(construct_c_case_2(num, den, x, pole, mul))
    return c