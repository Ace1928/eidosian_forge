from itertools import islice
from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import (collect, powsimp,  # type: ignore
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve
from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol
def lie_heuristic_abaco1_simple(match, comp=False):
    """
    The first heuristic uses the following four sets of
    assumptions on `\\xi` and `\\eta`

    .. math:: \\xi = 0, \\eta = f(x)

    .. math:: \\xi = 0, \\eta = f(y)

    .. math:: \\xi = f(x), \\eta = 0

    .. math:: \\xi = f(y), \\eta = 0

    The success of this heuristic is determined by algebraic factorisation.
    For the first assumption `\\xi = 0` and `\\eta` to be a function of `x`, the PDE

    .. math:: \\frac{\\partial \\eta}{\\partial x} + (\\frac{\\partial \\eta}{\\partial y}
                - \\frac{\\partial \\xi}{\\partial x})*h
                - \\frac{\\partial \\xi}{\\partial y}*h^{2}
                - \\xi*\\frac{\\partial h}{\\partial x} - \\eta*\\frac{\\partial h}{\\partial y} = 0

    reduces to `f'(x) - f\\frac{\\partial h}{\\partial y} = 0`
    If `\\frac{\\partial h}{\\partial y}` is a function of `x`, then this can usually
    be integrated easily. A similar idea is applied to the other 3 assumptions as well.


    References
    ==========

    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8


    """
    xieta = []
    y = match['y']
    h = match['h']
    func = match['func']
    x = func.args[0]
    hx = match['hx']
    hy = match['hy']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    hysym = hy.free_symbols
    if y not in hysym:
        try:
            fx = exp(integrate(hy, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fx}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    factor = hy / h
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: S.Zero, eta: fy.subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    factor = -hx / h
    facsym = factor.free_symbols
    if y not in facsym:
        try:
            fx = exp(integrate(factor, x))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fx, eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    factor = -hx / h ** 2
    facsym = factor.free_symbols
    if x not in facsym:
        try:
            fy = exp(integrate(factor, y))
        except NotImplementedError:
            pass
        else:
            inf = {xi: fy.subs(y, func), eta: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    if xieta:
        return xieta