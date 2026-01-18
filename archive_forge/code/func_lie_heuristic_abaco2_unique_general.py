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
def lie_heuristic_abaco2_unique_general(match, comp=False):
    """
    This heuristic finds if infinitesimals of the form `\\eta = f(x)`, `\\xi = g(y)`
    without making any assumptions on `h`.

    The complete sequence of steps is given in the paper mentioned below.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    A = hx.diff(y)
    B = hy.diff(y) + hy ** 2
    C = hx.diff(x) - hx ** 2
    if not (A and B and C):
        return
    Ax = A.diff(x)
    Ay = A.diff(y)
    Axy = Ax.diff(y)
    Axx = Ax.diff(x)
    Ayy = Ay.diff(y)
    D = simplify(2 * Axy + hx * Ay - Ax * hy + (hx * hy + 2 * A) * A) * A - 3 * Ax * Ay
    if not D:
        E1 = simplify(3 * Ax ** 2 + ((hx ** 2 + 2 * C) * A - 2 * Axx) * A)
        if E1:
            E2 = simplify((2 * Ayy + (2 * B - hy ** 2) * A) * A - 3 * Ay ** 2)
            if not E2:
                E3 = simplify(E1 * ((28 * Ax + 4 * hx * A) * A ** 3 - E1 * (hy * A + Ay)) - E1.diff(x) * 8 * A ** 4)
                if not E3:
                    etaval = cancel((4 * A ** 3 * (Ax - hx * A) + E1 * (hy * A - Ay)) / (S(2) * A * E1))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -4 * A ** 3 * etaval / E1
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]
    else:
        E1 = simplify((2 * Ayy + (2 * B - hy ** 2) * A) * A - 3 * Ay ** 2)
        if E1:
            E2 = simplify(4 * A ** 3 * D - D ** 2 + E1 * ((2 * Axx - (hx ** 2 + 2 * C) * A) * A - 3 * Ax ** 2))
            if not E2:
                E3 = simplify(-(A * D) * E1.diff(y) + ((E1.diff(x) - hy * D) * A + 3 * Ay * D + (A * hx - 3 * Ax) * E1) * E1)
                if not E3:
                    etaval = cancel(((A * hx - Ax) * E1 - (Ay + A * hy) * D) / (S(2) * A * D))
                    if x not in etaval:
                        try:
                            etaval = exp(integrate(etaval, y))
                        except NotImplementedError:
                            pass
                        else:
                            xival = -E1 * etaval / D
                            if y not in xival:
                                return [{xi: xival, eta: etaval.subs(y, func)}]