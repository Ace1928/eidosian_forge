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
def ode_2nd_power_series_regular(eq, func, order, match):
    """
    Gives a power series solution to a second order homogeneous differential
    equation with polynomial coefficients at a regular point. A second order
    homogeneous differential equation is of the form

    .. math :: P(x)\\frac{d^2y}{dx^2} + Q(x)\\frac{dy}{dx} + R(x) y(x) = 0

    A point is said to regular singular at `x0` if `x - x0\\frac{Q(x)}{P(x)}`
    and `(x - x0)^{2}\\frac{R(x)}{P(x)}` are analytic at `x0`. For simplicity
    `P(x)`, `Q(x)` and `R(x)` are assumed to be polynomials. The algorithm for
    finding the power series solutions is:

    1.  Try expressing `(x - x0)P(x)` and `((x - x0)^{2})Q(x)` as power series
        solutions about x0. Find `p0` and `q0` which are the constants of the
        power series expansions.
    2.  Solve the indicial equation `f(m) = m(m - 1) + m*p0 + q0`, to obtain the
        roots `m1` and `m2` of the indicial equation.
    3.  If `m1 - m2` is a non integer there exists two series solutions. If
        `m1 = m2`, there exists only one solution. If `m1 - m2` is an integer,
        then the existence of one solution is confirmed. The other solution may
        or may not exist.

    The power series solution is of the form `x^{m}\\sum_{n=0}^\\infty a_{n}x^{n}`. The
    coefficients are determined by the following recurrence relation.
    `a_{n} = -\\frac{\\sum_{k=0}^{n-1} q_{n-k} + (m + k)p_{n-k}}{f(m + n)}`. For the case
    in which `m1 - m2` is an integer, it can be seen from the recurrence relation
    that for the lower root `m`, when `n` equals the difference of both the
    roots, the denominator becomes zero. So if the numerator is not equal to zero,
    a second series solution exists.


    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = x*(f(x).diff(x, 2)) + 2*(f(x).diff(x)) + x*f(x)
    >>> pprint(dsolve(eq, hint='2nd_power_series_regular'))
                                  /    6    4    2    \\
                                  |   x    x    x     |
              /  4    2    \\   C1*|- --- + -- - -- + 1|
              | x    x     |      \\  720   24   2     /    / 6\\
    f(x) = C2*|--- - -- + 1| + ------------------------ + O\\x /
              \\120   6     /              x


    References
    ==========
    - George E. Simmons, "Differential Equations with Applications and
      Historical Notes", p.p 176 - 184

    """
    x = func.args[0]
    f = func.func
    C0, C1 = get_numbered_constants(eq, num=2)
    m = Dummy('m')
    x0 = match['x0']
    terms = match['terms']
    p = match['p']
    q = match['q']
    indicial = []
    for term in [p, q]:
        if not term.has(x):
            indicial.append(term)
        else:
            term = series(term, x=x, n=1, x0=x0)
            if isinstance(term, Order):
                indicial.append(S.Zero)
            else:
                for arg in term.args:
                    if not arg.has(x):
                        indicial.append(arg)
                        break
    p0, q0 = indicial
    sollist = solve(m * (m - 1) + m * p0 + q0, m)
    if sollist and isinstance(sollist, list) and all((sol.is_real for sol in sollist)):
        serdict1 = {}
        serdict2 = {}
        if len(sollist) == 1:
            m1 = m2 = sollist.pop()
            if terms - m1 - 1 <= 0:
                return Eq(f(x), Order(terms))
            serdict1 = _frobenius(terms - m1 - 1, m1, p0, q0, p, q, x0, x, C0)
        else:
            m1 = sollist[0]
            m2 = sollist[1]
            if m1 < m2:
                m1, m2 = (m2, m1)
            serdict1 = _frobenius(terms - m1 - 1, m1, p0, q0, p, q, x0, x, C0)
            if not (m1 - m2).is_integer:
                serdict2 = _frobenius(terms - m2 - 1, m2, p0, q0, p, q, x0, x, C1)
            else:
                serdict2 = _frobenius(terms - m2 - 1, m2, p0, q0, p, q, x0, x, C1, check=m1)
        if serdict1:
            finalseries1 = C0
            for key in serdict1:
                power = int(key.name[1:])
                finalseries1 += serdict1[key] * (x - x0) ** power
            finalseries1 = (x - x0) ** m1 * finalseries1
            finalseries2 = S.Zero
            if serdict2:
                for key in serdict2:
                    power = int(key.name[1:])
                    finalseries2 += serdict2[key] * (x - x0) ** power
                finalseries2 += C1
                finalseries2 = (x - x0) ** m2 * finalseries2
            return Eq(f(x), collect(finalseries1 + finalseries2, [C0, C1]) + Order(x ** terms))