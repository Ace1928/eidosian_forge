from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq
def bivariate_type(f, x, y, *, first=True):
    """Given an expression, f, 3 tests will be done to see what type
    of composite bivariate it might be, options for u(x, y) are::

        x*y
        x+y
        x*y+x
        x*y+y

    If it matches one of these types, ``u(x, y)``, ``P(u)`` and dummy
    variable ``u`` will be returned. Solving ``P(u)`` for ``u`` and
    equating the solutions to ``u(x, y)`` and then solving for ``x`` or
    ``y`` is equivalent to solving the original expression for ``x`` or
    ``y``. If ``x`` and ``y`` represent two functions in the same
    variable, e.g. ``x = g(t)`` and ``y = h(t)``, then if ``u(x, y) - p``
    can be solved for ``t`` then these represent the solutions to
    ``P(u) = 0`` when ``p`` are the solutions of ``P(u) = 0``.

    Only positive values of ``u`` are considered.

    Examples
    ========

    >>> from sympy import solve
    >>> from sympy.solvers.bivariate import bivariate_type
    >>> from sympy.abc import x, y
    >>> eq = (x**2 - 3).subs(x, x + y)
    >>> bivariate_type(eq, x, y)
    (x + y, _u**2 - 3, _u)
    >>> uxy, pu, u = _
    >>> usol = solve(pu, u); usol
    [sqrt(3)]
    >>> [solve(uxy - s) for s in solve(pu, u)]
    [[{x: -y + sqrt(3)}]]
    >>> all(eq.subs(s).equals(0) for sol in _ for s in sol)
    True

    """
    u = Dummy('u', positive=True)
    if first:
        p = Poly(f, x, y)
        f = p.as_expr()
        _x = Dummy()
        _y = Dummy()
        rv = bivariate_type(Poly(f.subs({x: _x, y: _y}), _x, _y), _x, _y, first=False)
        if rv:
            reps = {_x: x, _y: y}
            return (rv[0].xreplace(reps), rv[1].xreplace(reps), rv[2])
        return
    p = f
    f = p.as_expr()
    args = Add.make_args(p.as_expr())
    new = []
    for a in args:
        a = _mexpand(a.subs(x, u / y))
        free = a.free_symbols
        if x in free or y in free:
            break
        new.append(a)
    else:
        return (x * y, Add(*new), u)

    def ok(f, v, c):
        new = _mexpand(f.subs(v, c))
        free = new.free_symbols
        return None if x in free or y in free else new
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        a = root(p.coeff_monomial(x ** d), d)
        b = root(p.coeff_monomial(y ** d), d)
        new = ok(f, x, (u - b * y) / a)
        if new is not None:
            return (a * x + b * y, new, u)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        for itry in range(2):
            a = root(p.coeff_monomial(x ** d * y ** d), d)
            b = root(p.coeff_monomial(y ** d), d)
            new = ok(f, x, (u - b * y) / a / y)
            if new is not None:
                return (a * x * y + b * y, new, u)
            x, y = (y, x)