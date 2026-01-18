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
def construct_c_case_1(num, den, x, pole):
    num1, den1 = (num * Poly((x - pole) ** 2, x, extension=True)).cancel(den, include=True)
    r = num1.subs(x, pole) / den1.subs(x, pole)
    if r != -S(1) / 4:
        return [[(1 + sqrt(1 + 4 * r)) / 2], [(1 - sqrt(1 + 4 * r)) / 2]]
    return [[S.Half]]