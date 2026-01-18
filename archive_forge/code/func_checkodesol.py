from sympy.core import S, Pow
from sympy.core.function import (Derivative, AppliedUndef, diff)
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanAtom
from sympy.functions import exp
from sympy.series import Order
from sympy.simplify.simplify import simplify, posify, besselsimp
from sympy.simplify.trigsimp import trigsimp
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.solvers import solve
from sympy.solvers.deutils import _preprocess, ode_order
from sympy.utilities.iterables import iterable, is_sequence
def checkodesol(ode, sol, func=None, order='auto', solve_for_func=True):
    """
    Substitutes ``sol`` into ``ode`` and checks that the result is ``0``.

    This works when ``func`` is one function, like `f(x)` or a list of
    functions like `[f(x), g(x)]` when `ode` is a system of ODEs.  ``sol`` can
    be a single solution or a list of solutions.  Each solution may be an
    :py:class:`~sympy.core.relational.Equality` that the solution satisfies,
    e.g. ``Eq(f(x), C1), Eq(f(x) + C1, 0)``; or simply an
    :py:class:`~sympy.core.expr.Expr`, e.g. ``f(x) - C1``. In most cases it
    will not be necessary to explicitly identify the function, but if the
    function cannot be inferred from the original equation it can be supplied
    through the ``func`` argument.

    If a sequence of solutions is passed, the same sort of container will be
    used to return the result for each solution.

    It tries the following methods, in order, until it finds zero equivalence:

    1. Substitute the solution for `f` in the original equation.  This only
       works if ``ode`` is solved for `f`.  It will attempt to solve it first
       unless ``solve_for_func == False``.
    2. Take `n` derivatives of the solution, where `n` is the order of
       ``ode``, and check to see if that is equal to the solution.  This only
       works on exact ODEs.
    3. Take the 1st, 2nd, ..., `n`\\th derivatives of the solution, each time
       solving for the derivative of `f` of that order (this will always be
       possible because `f` is a linear operator). Then back substitute each
       derivative into ``ode`` in reverse order.

    This function returns a tuple.  The first item in the tuple is ``True`` if
    the substitution results in ``0``, and ``False`` otherwise. The second
    item in the tuple is what the substitution results in.  It should always
    be ``0`` if the first item is ``True``. Sometimes this function will
    return ``False`` even when an expression is identically equal to ``0``.
    This happens when :py:meth:`~sympy.simplify.simplify.simplify` does not
    reduce the expression to ``0``.  If an expression returned by this
    function vanishes identically, then ``sol`` really is a solution to
    the ``ode``.

    If this function seems to hang, it is probably because of a hard
    simplification.

    To use this function to test, test the first item of the tuple.

    Examples
    ========

    >>> from sympy import (Eq, Function, checkodesol, symbols,
    ...     Derivative, exp)
    >>> x, C1, C2 = symbols('x,C1,C2')
    >>> f, g = symbols('f g', cls=Function)
    >>> checkodesol(f(x).diff(x), Eq(f(x), C1))
    (True, 0)
    >>> assert checkodesol(f(x).diff(x), C1)[0]
    >>> assert not checkodesol(f(x).diff(x), x)[0]
    >>> checkodesol(f(x).diff(x, 2), x**2)
    (False, 2)

    >>> eqs = [Eq(Derivative(f(x), x), f(x)), Eq(Derivative(g(x), x), g(x))]
    >>> sol = [Eq(f(x), C1*exp(x)), Eq(g(x), C2*exp(x))]
    >>> checkodesol(eqs, sol)
    (True, [0, 0])

    """
    if iterable(ode):
        return checksysodesol(ode, sol, func=func)
    if not isinstance(ode, Equality):
        ode = Eq(ode, 0)
    if func is None:
        try:
            _, func = _preprocess(ode.lhs)
        except ValueError:
            funcs = [s.atoms(AppliedUndef) for s in (sol if is_sequence(sol, set) else [sol])]
            funcs = set().union(*funcs)
            if len(funcs) != 1:
                raise ValueError('must pass func arg to checkodesol for this case.')
            func = funcs.pop()
    if not isinstance(func, AppliedUndef) or len(func.args) != 1:
        raise ValueError('func must be a function of one variable, not %s' % func)
    if is_sequence(sol, set):
        return type(sol)([checkodesol(ode, i, order=order, solve_for_func=solve_for_func) for i in sol])
    if not isinstance(sol, Equality):
        sol = Eq(func, sol)
    elif sol.rhs == func:
        sol = sol.reversed
    if order == 'auto':
        order = ode_order(ode, func)
    solved = sol.lhs == func and (not sol.rhs.has(func))
    if solve_for_func and (not solved):
        rhs = solve(sol, func)
        if rhs:
            eqs = [Eq(func, t) for t in rhs]
            if len(rhs) == 1:
                eqs = eqs[0]
            return checkodesol(ode, eqs, order=order, solve_for_func=False)
    x = func.args[0]
    if sol.has(Order):
        assert sol.lhs == func
        Oterm = sol.rhs.getO()
        solrhs = sol.rhs.removeO()
        Oexpr = Oterm.expr
        assert isinstance(Oexpr, Pow)
        sorder = Oexpr.exp
        assert Oterm == Order(x ** sorder)
        odesubs = (ode.lhs - ode.rhs).subs(func, solrhs).doit().expand()
        neworder = Order(x ** (sorder - order))
        odesubs = odesubs + neworder
        assert odesubs.getO() == neworder
        residual = odesubs.removeO()
        return (residual == 0, residual)
    s = True
    testnum = 0
    while s:
        if testnum == 0:
            ode_diff = ode.lhs - ode.rhs
            if sol.lhs == func:
                s = sub_func_doit(ode_diff, func, sol.rhs)
                s = besselsimp(s)
            else:
                testnum += 1
                continue
            ss = simplify(s.rewrite(exp))
            if ss:
                s = ss.expand(force=True)
            else:
                s = 0
            testnum += 1
        elif testnum == 1:
            s = simplify(trigsimp(diff(sol.lhs, x, order) - diff(sol.rhs, x, order)) - trigsimp(ode.lhs) + trigsimp(ode.rhs))
            testnum += 1
        elif testnum == 2:
            if sol.lhs == func and (not sol.rhs.has(func)):
                diffsols = {0: sol.rhs}
            elif sol.rhs == func and (not sol.lhs.has(func)):
                diffsols = {0: sol.lhs}
            else:
                diffsols = {}
            sol = sol.lhs - sol.rhs
            for i in range(1, order + 1):
                if i == 1:
                    ds = sol.diff(x)
                    try:
                        sdf = solve(ds, func.diff(x, i))
                        if not sdf:
                            raise NotImplementedError
                    except NotImplementedError:
                        testnum += 1
                        break
                    else:
                        diffsols[i] = sdf[0]
                else:
                    diffsols[i] = diffsols[i - 1].diff(x)
            if testnum > 2:
                continue
            else:
                lhs, rhs = (ode.lhs, ode.rhs)
                for i in range(order, -1, -1):
                    if i == 0 and 0 not in diffsols:
                        break
                    lhs = sub_func_doit(lhs, func.diff(x, i), diffsols[i])
                    rhs = sub_func_doit(rhs, func.diff(x, i), diffsols[i])
                    ode_or_bool = Eq(lhs, rhs)
                    ode_or_bool = simplify(ode_or_bool)
                    if isinstance(ode_or_bool, (bool, BooleanAtom)):
                        if ode_or_bool:
                            lhs = rhs = S.Zero
                    else:
                        lhs = ode_or_bool.lhs
                        rhs = ode_or_bool.rhs
                num = trigsimp((lhs - rhs).as_numer_denom()[0])
                _func = Dummy('func')
                num = num.subs(func, _func)
                num, reps = posify(num)
                s = simplify(num).xreplace(reps).xreplace({_func: func})
                testnum += 1
        else:
            break
    if not s:
        return (True, s)
    elif s is True:
        raise NotImplementedError('Unable to test if ' + str(sol) + ' is a solution to ' + str(ode) + '.')
    else:
        return (False, s)