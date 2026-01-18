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
def ode_2nd_power_series_ordinary(eq, func, order, match):
    """
    Gives a power series solution to a second order homogeneous differential
    equation with polynomial coefficients at an ordinary point. A homogeneous
    differential equation is of the form

    .. math :: P(x)\\frac{d^2y}{dx^2} + Q(x)\\frac{dy}{dx} + R(x) y(x) = 0

    For simplicity it is assumed that `P(x)`, `Q(x)` and `R(x)` are polynomials,
    it is sufficient that `\\frac{Q(x)}{P(x)}` and `\\frac{R(x)}{P(x)}` exists at
    `x_{0}`. A recurrence relation is obtained by substituting `y` as `\\sum_{n=0}^\\infty a_{n}x^{n}`,
    in the differential equation, and equating the nth term. Using this relation
    various terms can be generated.


    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) + f(x)
    >>> pprint(dsolve(eq, hint='2nd_power_series_ordinary'))
              / 4    2    \\        /     2\\
              |x    x     |        |    x |    / 6\\
    f(x) = C2*|-- - -- + 1| + C1*x*|1 - --| + O\\x /
              \\24   2     /        \\    6 /


    References
    ==========
    - https://tutorial.math.lamar.edu/Classes/DE/SeriesSolutions.aspx
    - George E. Simmons, "Differential Equations with Applications and
      Historical Notes", p.p 176 - 184

    """
    x = func.args[0]
    f = func.func
    C0, C1 = get_numbered_constants(eq, num=2)
    n = Dummy('n', integer=True)
    s = Wild('s')
    k = Wild('k', exclude=[x])
    x0 = match['x0']
    terms = match['terms']
    p = match[match['a3']]
    q = match[match['b3']]
    r = match[match['c3']]
    seriesdict = {}
    recurr = Function('r')
    coefflist = [(recurr(n), r), (n * recurr(n), q), (n * (n - 1) * recurr(n), p)]
    for index, coeff in enumerate(coefflist):
        if coeff[1]:
            f2 = powsimp(expand((coeff[1] * (x - x0) ** (n - index)).subs(x, x + x0)))
            if f2.is_Add:
                addargs = f2.args
            else:
                addargs = [f2]
            for arg in addargs:
                powm = arg.match(s * x ** k)
                term = coeff[0] * powm[s]
                if not powm[k].is_Symbol:
                    term = term.subs(n, n - powm[k].as_independent(n)[0])
                startind = powm[k].subs(n, index)
                if startind:
                    for i in reversed(range(startind)):
                        if not term.subs(n, i):
                            seriesdict[term] = i
                        else:
                            seriesdict[term] = i + 1
                            break
                else:
                    seriesdict[term] = S.Zero
    teq = S.Zero
    suminit = seriesdict.values()
    rkeys = seriesdict.keys()
    req = Add(*rkeys)
    if any(suminit):
        maxval = max(suminit)
        for term in seriesdict:
            val = seriesdict[term]
            if val != maxval:
                for i in range(val, maxval):
                    teq += term.subs(n, val)
    finaldict = {}
    if teq:
        fargs = teq.atoms(AppliedUndef)
        if len(fargs) == 1:
            finaldict[fargs.pop()] = 0
        else:
            maxf = max(fargs, key=lambda x: x.args[0])
            sol = solve(teq, maxf)
            if isinstance(sol, list):
                sol = sol[0]
            finaldict[maxf] = sol
    fargs = req.atoms(AppliedUndef)
    maxf = max(fargs, key=lambda x: x.args[0])
    minf = min(fargs, key=lambda x: x.args[0])
    if minf.args[0].is_Symbol:
        startiter = 0
    else:
        startiter = -minf.args[0].as_independent(n)[0]
    lhs = maxf
    rhs = solve(req, maxf)
    if isinstance(rhs, list):
        rhs = rhs[0]
    tcounter = len([t for t in finaldict.values() if t])
    for _ in range(tcounter, terms - 3):
        check = rhs.subs(n, startiter)
        nlhs = lhs.subs(n, startiter)
        nrhs = check.subs(finaldict)
        finaldict[nlhs] = nrhs
        startiter += 1
    series = C0 + C1 * (x - x0)
    for term in finaldict:
        if finaldict[term]:
            fact = term.args[0]
            series += finaldict[term].subs([(recurr(0), C0), (recurr(1), C1)]) * (x - x0) ** fact
    series = collect(expand_mul(series), [C0, C1]) + Order(x ** terms)
    return Eq(f(x), series)