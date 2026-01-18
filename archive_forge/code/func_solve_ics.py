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
def solve_ics(sols, funcs, constants, ics):
    """
    Solve for the constants given initial conditions

    ``sols`` is a list of solutions.

    ``funcs`` is a list of functions.

    ``constants`` is a list of constants.

    ``ics`` is the set of initial/boundary conditions for the differential
    equation. It should be given in the form of ``{f(x0): x1,
    f(x).diff(x).subs(x, x2):  x3}`` and so on.

    Returns a dictionary mapping constants to values.
    ``solution.subs(constants)`` will replace the constants in ``solution``.

    Example
    =======
    >>> # From dsolve(f(x).diff(x) - f(x), f(x))
    >>> from sympy import symbols, Eq, exp, Function
    >>> from sympy.solvers.ode.ode import solve_ics
    >>> f = Function('f')
    >>> x, C1 = symbols('x C1')
    >>> sols = [Eq(f(x), C1*exp(x))]
    >>> funcs = [f(x)]
    >>> constants = [C1]
    >>> ics = {f(0): 2}
    >>> solved_constants = solve_ics(sols, funcs, constants, ics)
    >>> solved_constants
    {C1: 2}
    >>> sols[0].subs(solved_constants)
    Eq(f(x), 2*exp(x))

    """
    x = funcs[0].args[0]
    diff_sols = []
    subs_sols = []
    diff_variables = set()
    for funcarg, value in ics.items():
        if isinstance(funcarg, AppliedUndef):
            x0 = funcarg.args[0]
            matching_func = [f for f in funcs if f.func == funcarg.func][0]
            S = sols
        elif isinstance(funcarg, (Subs, Derivative)):
            if isinstance(funcarg, Subs):
                funcarg = funcarg.doit()
            if isinstance(funcarg, Subs):
                deriv = funcarg.expr
                x0 = funcarg.point[0]
                variables = funcarg.expr.variables
                matching_func = deriv
            elif isinstance(funcarg, Derivative):
                deriv = funcarg
                x0 = funcarg.variables[0]
                variables = (x,) * len(funcarg.variables)
                matching_func = deriv.subs(x0, x)
            for sol in sols:
                if sol.has(deriv.expr.func):
                    diff_sols.append(Eq(sol.lhs.diff(*variables), sol.rhs.diff(*variables)))
            diff_variables.add(variables)
            S = diff_sols
        else:
            raise NotImplementedError('Unrecognized initial condition')
        for sol in S:
            if sol.has(matching_func):
                sol2 = sol
                sol2 = sol2.subs(x, x0)
                sol2 = sol2.subs(funcarg, value)
                if not isinstance(sol2, BooleanAtom) or not subs_sols:
                    subs_sols = [s for s in subs_sols if not isinstance(s, BooleanAtom)]
                    subs_sols.append(sol2)
    try:
        solved_constants = solve(subs_sols, constants, dict=True)
    except NotImplementedError:
        solved_constants = []
    if not solved_constants:
        raise ValueError("Couldn't solve for initial conditions")
    if solved_constants == True:
        raise ValueError('Initial conditions did not produce any solutions for constants. Perhaps they are degenerate.')
    if len(solved_constants) > 1:
        raise NotImplementedError('Initial conditions produced too many solutions for constants')
    return solved_constants[0]