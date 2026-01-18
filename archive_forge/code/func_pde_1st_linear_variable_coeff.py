from functools import reduce
from itertools import combinations_with_replacement
from sympy.simplify import simplify  # type: ignore
from sympy.core import Add, S
from sympy.core.function import Function, expand, AppliedUndef, Subs
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, symbols
from sympy.functions import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.utilities.iterables import has_dups, is_sequence
from sympy.utilities.misc import filldedent
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from sympy.solvers.solvers import solve
from sympy.simplify.radsimp import collect
import operator
def pde_1st_linear_variable_coeff(eq, func, order, match, solvefun):
    """
    Solves a first order linear partial differential equation
    with variable coefficients. The general form of this partial
    differential equation is

    .. math:: a(x, y) \\frac{\\partial f(x, y)}{\\partial x}
                + b(x, y) \\frac{\\partial f(x, y)}{\\partial y}
                + c(x, y) f(x, y) = G(x, y)

    where `a(x, y)`, `b(x, y)`, `c(x, y)` and `G(x, y)` are arbitrary
    functions in `x` and `y`. This PDE is converted into an ODE by
    making the following transformation:

    1. `\\xi` as `x`

    2. `\\eta` as the constant in the solution to the differential
       equation `\\frac{dy}{dx} = -\\frac{b}{a}`

    Making the previous substitutions reduces it to the linear ODE

    .. math:: a(\\xi, \\eta)\\frac{du}{d\\xi} + c(\\xi, \\eta)u - G(\\xi, \\eta) = 0

    which can be solved using ``dsolve``.

    >>> from sympy.abc import x, y
    >>> from sympy import Function, pprint
    >>> a, b, c, G, f= [Function(i) for i in ['a', 'b', 'c', 'G', 'f']]
    >>> u = f(x,y)
    >>> ux = u.diff(x)
    >>> uy = u.diff(y)
    >>> genform = a(x, y)*u + b(x, y)*ux + c(x, y)*uy - G(x,y)
    >>> pprint(genform)
                                         d                     d
    -G(x, y) + a(x, y)*f(x, y) + b(x, y)*--(f(x, y)) + c(x, y)*--(f(x, y))
                                         dx                    dy


    Examples
    ========

    >>> from sympy.solvers.pde import pdsolve
    >>> from sympy import Function, pprint
    >>> from sympy.abc import x,y
    >>> f = Function('f')
    >>> eq =  x*(u.diff(x)) - y*(u.diff(y)) + y**2*u - y**2
    >>> pdsolve(eq)
    Eq(f(x, y), F(x*y)*exp(y**2/2) + 1)

    References
    ==========

    - Viktor Grigoryan, "Partial Differential Equations"
      Math 124A - Fall 2010, pp.7

    """
    from sympy.solvers.ode import dsolve
    xi, eta = symbols('xi eta')
    f = func.func
    x = func.args[0]
    y = func.args[1]
    b = match[match['b']]
    c = match[match['c']]
    d = match[match['d']]
    e = -match[match['e']]
    if not d:
        if not (b and c):
            if c:
                try:
                    tsol = integrate(e / c, y)
                except NotImplementedError:
                    raise NotImplementedError('Unable to find a solution due to inability of integrate')
                else:
                    return Eq(f(x, y), solvefun(x) + tsol)
            if b:
                try:
                    tsol = integrate(e / b, x)
                except NotImplementedError:
                    raise NotImplementedError('Unable to find a solution due to inability of integrate')
                else:
                    return Eq(f(x, y), solvefun(y) + tsol)
    if not c:
        plode = f(x).diff(x) * b + d * f(x) - e
        sol = dsolve(plode, f(x))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, y)
        return Eq(f(x, y), rhs)
    if not b:
        plode = f(y).diff(y) * c + d * f(y) - e
        sol = dsolve(plode, f(y))
        syms = sol.free_symbols - plode.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(sol.rhs, syms, solvefun, x)
        return Eq(f(x, y), rhs)
    dummy = Function('d')
    h = (c / b).subs(y, dummy(x))
    sol = dsolve(dummy(x).diff(x) - h, dummy(x))
    if isinstance(sol, list):
        sol = sol[0]
    solsym = sol.free_symbols - h.free_symbols - {x, y}
    if len(solsym) == 1:
        solsym = solsym.pop()
        etat = solve(sol, solsym)[0].subs(dummy(x), y)
        ysub = solve(eta - etat, y)[0]
        deq = (b * f(x).diff(x) + d * f(x) - e).subs(y, ysub)
        final = dsolve(deq, f(x), hint='1st_linear').rhs
        if isinstance(final, list):
            final = final[0]
        finsyms = final.free_symbols - deq.free_symbols - {x, y}
        rhs = _simplify_variable_coeff(final, finsyms, solvefun, etat)
        return Eq(f(x, y), rhs)
    else:
        raise NotImplementedError('Cannot solve the partial differential equation due to inability of constantsimp')