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
def lie_heuristic_abaco2_unique_unknown(match, comp=False):
    """
    This heuristic assumes the presence of unknown functions or known functions
    with non-integer powers.

    1. A list of all functions and non-integer powers containing x and y
    2. Loop over each element `f` in the list, find `\\frac{\\frac{\\partial f}{\\partial x}}{
       \\frac{\\partial f}{\\partial x}} = R`

       If it is separable in `x` and `y`, let `X` be the factors containing `x`. Then

       a] Check if `\\xi = X` and `\\eta = -\\frac{X}{R}` satisfy the PDE. If yes, then return
          `\\xi` and `\\eta`
       b] Check if `\\xi = \\frac{-R}{X}` and `\\eta = -\\frac{1}{X}` satisfy the PDE.
           If yes, then return `\\xi` and `\\eta`

       If not, then check if

       a] :math:`\\xi = -R,\\eta = 1`

       b] :math:`\\xi = 1, \\eta = -\\frac{1}{R}`

       are solutions.

    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 10 - pp. 12

    """
    h = match['h']
    hx = match['hx']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    funclist = []
    for atom in h.atoms(Pow):
        base, exp = atom.as_base_exp()
        if base.has(x) and base.has(y):
            if not exp.is_Integer:
                funclist.append(atom)
    for function in h.atoms(AppliedUndef):
        syms = function.free_symbols
        if x in syms and y in syms:
            funclist.append(function)
    for f in funclist:
        frac = cancel(f.diff(y) / f.diff(x))
        sep = separatevars(frac, dict=True, symbols=[x, y])
        if sep and sep['coeff']:
            xitry1 = sep[x]
            etatry1 = -1 / (sep[y] * sep['coeff'])
            pde1 = etatry1.diff(y) * h - xitry1.diff(x) * h - xitry1 * hx - etatry1 * hy
            if not simplify(pde1):
                return [{xi: xitry1, eta: etatry1.subs(y, func)}]
            xitry2 = 1 / etatry1
            etatry2 = 1 / xitry1
            pde2 = etatry2.diff(x) - xitry2.diff(y) * h ** 2 - xitry2 * hx - etatry2 * hy
            if not simplify(expand(pde2)):
                return [{xi: xitry2.subs(y, func), eta: etatry2}]
        else:
            etatry = -1 / frac
            pde = etatry.diff(x) + etatry.diff(y) * h - hx - etatry * hy
            if not simplify(pde):
                return [{xi: S.One, eta: etatry.subs(y, func)}]
            xitry = -frac
            pde = -xitry.diff(x) * h - xitry.diff(y) * h ** 2 - xitry * hx - hy
            if not simplify(expand(pde)):
                return [{xi: xitry.subs(y, func), eta: S.One}]