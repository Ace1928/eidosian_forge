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
def homogeneous_order(eq, *symbols):
    """
    Returns the order `n` if `g` is homogeneous and ``None`` if it is not
    homogeneous.

    Determines if a function is homogeneous and if so of what order.  A
    function `f(x, y, \\cdots)` is homogeneous of order `n` if `f(t x, t y,
    \\cdots) = t^n f(x, y, \\cdots)`.

    If the function is of two variables, `F(x, y)`, then `f` being homogeneous
    of any order is equivalent to being able to rewrite `F(x, y)` as `G(x/y)`
    or `H(y/x)`.  This fact is used to solve 1st order ordinary differential
    equations whose coefficients are homogeneous of the same order (see the
    docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`).

    Symbols can be functions, but every argument of the function must be a
    symbol, and the arguments of the function that appear in the expression
    must match those given in the list of symbols.  If a declared function
    appears with different arguments than given in the list of symbols,
    ``None`` is returned.

    Examples
    ========

    >>> from sympy import Function, homogeneous_order, sqrt
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> homogeneous_order(f(x), f(x)) is None
    True
    >>> homogeneous_order(f(x,y), f(y, x), x, y) is None
    True
    >>> homogeneous_order(f(x), f(x), x)
    1
    >>> homogeneous_order(x**2*f(x)/sqrt(x**2+f(x)**2), x, f(x))
    2
    >>> homogeneous_order(x**2+f(x), x, f(x)) is None
    True

    """
    if not symbols:
        raise ValueError('homogeneous_order: no symbols were given.')
    symset = set(symbols)
    eq = sympify(eq)
    if eq.has(Order, Derivative):
        return None
    if eq.is_Number or eq.is_NumberSymbol or eq.is_number:
        return S.Zero
    dum = numbered_symbols(prefix='d', cls=Dummy)
    newsyms = set()
    for i in [j for j in symset if getattr(j, 'is_Function')]:
        iargs = set(i.args)
        if iargs.difference(symset):
            return None
        else:
            dummyvar = next(dum)
            eq = eq.subs(i, dummyvar)
            symset.remove(i)
            newsyms.add(dummyvar)
    symset.update(newsyms)
    if not eq.free_symbols & symset:
        return None
    if isinstance(eq, Function):
        return None if homogeneous_order(eq.args[0], *tuple(symset)) != 0 else S.Zero
    t = Dummy('t', positive=True)
    eqs = separatevars(eq.subs([(i, t * i) for i in symset]), [t], dict=True)[t]
    if eqs is S.One:
        return S.Zero
    i, d = eqs.as_independent(t, as_Add=False)
    b, e = d.as_base_exp()
    if b == t:
        return e