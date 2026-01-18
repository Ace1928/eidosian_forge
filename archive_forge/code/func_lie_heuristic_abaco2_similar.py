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
def lie_heuristic_abaco2_similar(match, comp=False):
    """
    This heuristic uses the following two assumptions on `\\xi` and `\\eta`

    .. math:: \\eta = g(x), \\xi = f(x)

    .. math:: \\eta = f(y), \\xi = g(y)

    For the first assumption,

    1. First `\\frac{\\frac{\\partial h}{\\partial y}}{\\frac{\\partial^{2} h}{
       \\partial yy}}` is calculated. Let us say this value is A

    2. If this is constant, then `h` is matched to the form `A(x) + B(x)e^{
       \\frac{y}{C}}` then, `\\frac{e^{\\int \\frac{A(x)}{C} \\,dx}}{B(x)}` gives `f(x)`
       and `A(x)*f(x)` gives `g(x)`

    3. Otherwise `\\frac{\\frac{\\partial A}{\\partial X}}{\\frac{\\partial A}{
       \\partial Y}} = \\gamma` is calculated. If

       a] `\\gamma` is a function of `x` alone

       b] `\\frac{\\gamma\\frac{\\partial h}{\\partial y} - \\gamma'(x) - \\frac{
       \\partial h}{\\partial x}}{h + \\gamma} = G` is a function of `x` alone.
       then, `e^{\\int G \\,dx}` gives `f(x)` and `-\\gamma*f(x)` gives `g(x)`

    The second assumption holds good if `\\frac{dy}{dx} = h(x, y)` is rewritten as
    `\\frac{dy}{dx} = \\frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(x)`, the coordinates are again
    interchanged, to get `\\xi` as `f(x^*)` and `\\eta` as `g(y^*)`

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    hinv = match['hinv']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    factor = cancel(h.diff(y) / h.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and (not factor.has(y)):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B * exp(y / C))
        try:
            tau = exp(-integrate(match[A] / match[C]), x) / match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A] * tau
            return [{xi: tau, eta: gx}]
    else:
        gamma = cancel(factorx / factory)
        if not gamma.has(y):
            tauint = cancel((gamma * hy - gamma.diff(x) - hx) / (h + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau * gamma
                    return [{xi: tau, eta: gx}]
    factor = cancel(hinv.diff(y) / hinv.diff(y, 2))
    factorx = factor.diff(x)
    factory = factor.diff(y)
    if not factor.has(x) and (not factor.has(y)):
        A = Wild('A', exclude=[y])
        B = Wild('B', exclude=[y])
        C = Wild('C', exclude=[x, y])
        match = h.match(A + B * exp(y / C))
        try:
            tau = exp(-integrate(match[A] / match[C]), x) / match[B]
        except NotImplementedError:
            pass
        else:
            gx = match[A] * tau
            return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]
    else:
        gamma = cancel(factorx / factory)
        if not gamma.has(y):
            tauint = cancel((gamma * hinv.diff(y) - gamma.diff(x) - hinv.diff(x)) / (hinv + gamma))
            if not tauint.has(y):
                try:
                    tau = exp(integrate(tauint, x))
                except NotImplementedError:
                    pass
                else:
                    gx = -tau * gamma
                    return [{eta: tau.subs(x, func), xi: gx.subs(x, func)}]