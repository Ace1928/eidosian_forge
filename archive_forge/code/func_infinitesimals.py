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
def infinitesimals(eq, func=None, order=None, hint='default', match=None):
    """
    The infinitesimal functions of an ordinary differential equation, `\\xi(x,y)`
    and `\\eta(x,y)`, are the infinitesimals of the Lie group of point transformations
    for which the differential equation is invariant. So, the ODE `y'=f(x,y)`
    would admit a Lie group `x^*=X(x,y;\\varepsilon)=x+\\varepsilon\\xi(x,y)`,
    `y^*=Y(x,y;\\varepsilon)=y+\\varepsilon\\eta(x,y)` such that `(y^*)'=f(x^*, y^*)`.
    A change of coordinates, to `r(x,y)` and `s(x,y)`, can be performed so this Lie group
    becomes the translation group, `r^*=r` and `s^*=s+\\varepsilon`.
    They are tangents to the coordinate curves of the new system.

    Consider the transformation `(x, y) \\to (X, Y)` such that the
    differential equation remains invariant. `\\xi` and `\\eta` are the tangents to
    the transformed coordinates `X` and `Y`, at `\\varepsilon=0`.

    .. math:: \\left(\\frac{\\partial X(x,y;\\varepsilon)}{\\partial\\varepsilon
                }\\right)|_{\\varepsilon=0} = \\xi,
              \\left(\\frac{\\partial Y(x,y;\\varepsilon)}{\\partial\\varepsilon
                }\\right)|_{\\varepsilon=0} = \\eta,

    The infinitesimals can be found by solving the following PDE:

        >>> from sympy import Function, Eq, pprint
        >>> from sympy.abc import x, y
        >>> xi, eta, h = map(Function, ['xi', 'eta', 'h'])
        >>> h = h(x, y)  # dy/dx = h
        >>> eta = eta(x, y)
        >>> xi = xi(x, y)
        >>> genform = Eq(eta.diff(x) + (eta.diff(y) - xi.diff(x))*h
        ... - (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)), 0)
        >>> pprint(genform)
        /d               d           \\                     d              2       d
        |--(eta(x, y)) - --(xi(x, y))|*h(x, y) - eta(x, y)*--(h(x, y)) - h (x, y)*--(x
        \\dy              dx          /                     dy                     dy
        <BLANKLINE>
                            d             d
        i(x, y)) - xi(x, y)*--(h(x, y)) + --(eta(x, y)) = 0
                            dx            dx

    Solving the above mentioned PDE is not trivial, and can be solved only by
    making intelligent assumptions for `\\xi` and `\\eta` (heuristics). Once an
    infinitesimal is found, the attempt to find more heuristics stops. This is done to
    optimise the speed of solving the differential equation. If a list of all the
    infinitesimals is needed, ``hint`` should be flagged as ``all``, which gives
    the complete list of infinitesimals. If the infinitesimals for a particular
    heuristic needs to be found, it can be passed as a flag to ``hint``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.ode.lie_group import infinitesimals
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x) - x**2*f(x)
    >>> infinitesimals(eq)
    [{eta(x, f(x)): exp(x**3/3), xi(x, f(x)): 0}]

    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        eq, func = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError("Infinitesimals for only first order ODE's have been implemented")
        else:
            df = func.diff(x)
            a = Wild('a', exclude=[df])
            b = Wild('b', exclude=[df])
            if match:
                h = match['h']
                y = match['y']
            else:
                match = collect(expand(eq), df).match(a * df + b)
                if match:
                    h = -simplify(match[b] / match[a])
                else:
                    try:
                        sol = solve(eq, df)
                    except NotImplementedError:
                        raise NotImplementedError('Infinitesimals for the first order ODE could not be found')
                    else:
                        h = sol[0]
                y = Dummy('y')
                h = h.subs(func, y)
            u = Dummy('u')
            hx = h.diff(x)
            hy = h.diff(y)
            hinv = (1 / h).subs([(x, u), (y, x)]).subs(u, y)
            match = {'h': h, 'func': func, 'hx': hx, 'hy': hy, 'y': y, 'hinv': hinv}
            if hint == 'all':
                xieta = []
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    inflist = function(match, comp=True)
                    if inflist:
                        xieta.extend([inf for inf in inflist if inf not in xieta])
                if xieta:
                    return xieta
                else:
                    raise NotImplementedError('Infinitesimals could not be found for the given ODE')
            elif hint == 'default':
                for heuristic in lie_heuristics:
                    function = globals()['lie_heuristic_' + heuristic]
                    xieta = function(match, comp=False)
                    if xieta:
                        return xieta
                raise NotImplementedError('Infinitesimals could not be found for the given ODE')
            elif hint not in lie_heuristics:
                raise ValueError('Heuristic not recognized: ' + hint)
            else:
                function = globals()['lie_heuristic_' + hint]
                xieta = function(match, comp=True)
                if xieta:
                    return xieta
                else:
                    raise ValueError('Infinitesimals could not be found using the given heuristic')