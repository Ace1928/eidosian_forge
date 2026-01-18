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
def lie_heuristic_function_sum(match, comp=False):
    """
    This heuristic uses the following two assumptions on `\\xi` and `\\eta`

    .. math:: \\eta = 0, \\xi = f(x) + g(y)

    .. math:: \\eta = f(x) + g(y), \\xi = 0

    The first assumption of this heuristic holds good if

    .. math:: \\frac{\\partial}{\\partial y}[(h\\frac{\\partial^{2}}{
                \\partial x^{2}}(h^{-1}))^{-1}]

    is separable in `x` and `y`,

    1. The separated factors containing `y` is `\\frac{\\partial g}{\\partial y}`.
       From this `g(y)` can be determined.
    2. The separated factors containing `x` is `f''(x)`.
    3. `h\\frac{\\partial^{2}}{\\partial x^{2}}(h^{-1})` equals
       `\\frac{f''(x)}{f(x) + g(y)}`. From this `f(x)` can be determined.

    The second assumption holds good if `\\frac{dy}{dx} = h(x, y)` is rewritten as
    `\\frac{dy}{dx} = \\frac{1}{h(y, x)}` and the same properties of the first
    assumption satisfies. After obtaining `f(x)` and `g(y)`, the coordinates
    are again interchanged, to get `\\eta` as `f(x) + g(y)`.

    For both assumptions, the constant factors are separated among `g(y)`
    and `f''(x)`, such that `f''(x)` obtained from 3] is the same as that
    obtained from 2]. If not possible, then this heuristic fails.


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """
    xieta = []
    h = match['h']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    for odefac in [h, hinv]:
        factor = odefac * (1 / odefac).diff(x, 2)
        sep = separatevars((1 / factor).diff(y), dict=True, symbols=[x, y])
        if sep and sep['coeff'] and sep[x].has(x) and sep[y].has(y):
            k = Dummy('k')
            try:
                gy = k * integrate(sep[y], y)
            except NotImplementedError:
                pass
            else:
                fdd = 1 / (k * sep[x] * sep['coeff'])
                fx = simplify(fdd / factor - gy)
                check = simplify(fx.diff(x, 2) - fdd)
                if fx:
                    if not check:
                        fx = fx.subs(k, 1)
                        gy = gy / k
                    else:
                        sol = solve(check, k)
                        if sol:
                            sol = sol[0]
                            fx = fx.subs(k, sol)
                            gy = gy / k * sol
                        else:
                            continue
                    if odefac == hinv:
                        fx = fx.subs(x, y)
                        gy = gy.subs(y, x)
                    etaval = factor_terms(fx + gy)
                    if etaval.is_Mul:
                        etaval = Mul(*[arg for arg in etaval.args if arg.has(x, y)])
                    if odefac == hinv:
                        inf = {eta: etaval.subs(y, func), xi: S.Zero}
                    else:
                        inf = {xi: etaval.subs(y, func), eta: S.Zero}
                    if not comp:
                        return [inf]
                    else:
                        xieta.append(inf)
        if xieta:
            return xieta