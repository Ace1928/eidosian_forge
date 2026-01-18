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
def lie_heuristic_bivariate(match, comp=False):
    """
    The third heuristic assumes the infinitesimals `\\xi` and `\\eta`
    to be bi-variate polynomials in `x` and `y`. The assumption made here
    for the logic below is that `h` is a rational function in `x` and `y`
    though that may not be necessary for the infinitesimals to be
    bivariate polynomials. The coefficients of the infinitesimals
    are found out by substituting them in the PDE and grouping similar terms
    that are polynomials and since they form a linear system, solve and check
    for non trivial solutions. The degree of the assumed bivariates
    are increased till a certain maximum value.

    References
    ==========
    - Lie Groups and Differential Equations
      pp. 327 - pp. 329

    """
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    if h.is_rational_function():
        etax, etay, etad, xix, xiy, xid = symbols('etax etay etad xix xiy xid')
        ipde = etax + (etay - xix) * h - xiy * h ** 2 - xid * hx - etad * hy
        num, denom = cancel(ipde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()
        deta = Function('deta')(x, y)
        dxi = Function('dxi')(x, y)
        ipde = deta.diff(x) + (deta.diff(y) - dxi.diff(x)) * h - dxi.diff(y) * h ** 2 - dxi * hx - deta * hy
        xieq = Symbol('xi0')
        etaeq = Symbol('eta0')
        for i in range(deg + 1):
            if i:
                xieq += Add(*[Symbol('xi_' + str(power) + '_' + str(i - power)) * x ** power * y ** (i - power) for power in range(i + 1)])
                etaeq += Add(*[Symbol('eta_' + str(power) + '_' + str(i - power)) * x ** power * y ** (i - power) for power in range(i + 1)])
            pden, denom = ipde.subs({dxi: xieq, deta: etaeq}).doit().as_numer_denom()
            pden = expand(pden)
            if pden.is_polynomial(x, y) and pden.is_Add:
                polyy = Poly(pden, x, y).as_dict()
            if polyy:
                symset = xieq.free_symbols.union(etaeq.free_symbols) - {x, y}
                soldict = solve(polyy.values(), *symset)
                if isinstance(soldict, list):
                    soldict = soldict[0]
                if any(soldict.values()):
                    xired = xieq.subs(soldict)
                    etared = etaeq.subs(soldict)
                    dict_ = {sym: 1 for sym in symset}
                    inf = {eta: etared.subs(dict_).subs(y, func), xi: xired.subs(dict_).subs(y, func)}
                    return [inf]