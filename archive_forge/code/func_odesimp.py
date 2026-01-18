from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve
from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from .single import SingleODEProblem, SingleODESolver, solver_map
@vectorize(0)
def odesimp(ode, eq, func, hint):
    """
    Simplifies solutions of ODEs, including trying to solve for ``func`` and
    running :py:meth:`~sympy.solvers.ode.constantsimp`.

    It may use knowledge of the type of solution that the hint returns to
    apply additional simplifications.

    It also attempts to integrate any :py:class:`~sympy.integrals.integrals.Integral`\\s
    in the expression, if the hint is not an ``_Integral`` hint.

    This function should have no effect on expressions returned by
    :py:meth:`~sympy.solvers.ode.dsolve`, as
    :py:meth:`~sympy.solvers.ode.dsolve` already calls
    :py:meth:`~sympy.solvers.ode.ode.odesimp`, but the individual hint functions
    do not call :py:meth:`~sympy.solvers.ode.ode.odesimp` (because the
    :py:meth:`~sympy.solvers.ode.dsolve` wrapper does).  Therefore, this
    function is designed for mainly internal use.

    Examples
    ========

    >>> from sympy import sin, symbols, dsolve, pprint, Function
    >>> from sympy.solvers.ode.ode import odesimp
    >>> x, u2, C1= symbols('x,u2,C1')
    >>> f = Function('f')

    >>> eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    ... simplify=False)
    >>> pprint(eq, wrap_line=False)
                            x
                           ----
                           f(x)
                             /
                            |
                            |   /        1   \\
                            |  -|u1 + -------|
                            |   |        /1 \\|
                            |   |     sin|--||
                            |   \\        \\u1//
    log(f(x)) = log(C1) +   |  ---------------- d(u1)
                            |          2
                            |        u1
                            |
                           /

    >>> pprint(odesimp(eq, f(x), 1, {C1},
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep'
    ... )) #doctest: +SKIP
        x
    --------- = C1
       /f(x)\\
    tan|----|
       \\2*x /

    """
    x = func.args[0]
    f = func.func
    C1 = get_numbered_constants(eq, num=1)
    constants = eq.free_symbols - ode.free_symbols
    eq = _handle_Integral(eq, func, hint)
    if hint.startswith('nth_linear_euler_eq_nonhomogeneous'):
        eq = simplify(eq)
    if not isinstance(eq, Equality):
        raise TypeError('eq should be an instance of Equality')
    eq = constantsimp(eq, constants)
    if eq.rhs == func and (not eq.lhs.has(func)):
        eq = [Eq(eq.rhs, eq.lhs)]
    if eq.lhs == func and (not eq.rhs.has(func)):
        eq = [eq]
    else:
        try:
            floats = any((i.is_Float for i in eq.atoms(Number)))
            eqsol = solve(eq, func, force=True, rational=False if floats else None)
            if not eqsol:
                raise NotImplementedError
        except (NotImplementedError, PolynomialError):
            eq = [eq]
        else:

            def _expand(expr):
                numer, denom = expr.as_numer_denom()
                if denom.is_Add:
                    return expr
                else:
                    return powsimp(expr.expand(), combine='exp', deep=True)
            eq = [Eq(f(x), _expand(t)) for t in eqsol]
        if hint.startswith('1st_homogeneous_coeff'):
            for j, eqi in enumerate(eq):
                newi = logcombine(eqi, force=True)
                if isinstance(newi.lhs, log) and newi.rhs == 0:
                    newi = Eq(newi.lhs.args[0] / C1, C1)
                eq[j] = newi
    for i, eqi in enumerate(eq):
        eq[i] = constantsimp(eqi, constants)
        eq[i] = constant_renumber(eq[i], ode.free_symbols)
    if len(eq) == 1:
        eq = eq[0]
    return eq