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
def get_gen_sol_from_part_sol(part_sols, a, x):
    """"
    Helper function which computes the general
    solution for a Riccati ODE from its particular
    solutions.

    There are 3 cases to find the general solution
    from the particular solutions for a Riccati ODE
    depending on the number of particular solution(s)
    we have - 1, 2 or 3.

    For more information, see Section 6 of
    "Methods of Solution of the Riccati Differential Equation"
    by D. R. Haaheim and F. M. Stein
    """
    if len(part_sols) == 0:
        return []
    elif len(part_sols) == 1:
        y1 = part_sols[0]
        i = exp(Integral(2 * y1, x))
        z = i * Integral(a / i, x)
        z = z.doit()
        if a == 0 or z == 0:
            return y1
        return y1 + 1 / z
    elif len(part_sols) == 2:
        y1, y2 = part_sols
        if len(y1.atoms(Dummy)) + len(y2.atoms(Dummy)) > 0:
            u = exp(Integral(y2 - y1, x)).doit()
        else:
            C1 = Dummy('C1')
            u = C1 * exp(Integral(y2 - y1, x)).doit()
        if u == 1:
            return y2
        return (y2 * u - y1) / (u - 1)
    else:
        y1, y2, y3 = part_sols[:3]
        C1 = Dummy('C1')
        return (C1 + 1) * y2 * (y1 - y3) / (C1 * y1 + y2 - (C1 + 1) * y3)