from __future__ import annotations
from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import ilcm, Float, Rational, _illegal
from sympy.core.power import integer_log, Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom
from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
from sympy.utilities.decorator import conserve_mpmath_dps
from mpmath import findroot
from sympy.solvers.polysys import solve_poly_system
from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product
import warnings
from sympy.solvers.bivariate import (
def solve_linear(lhs, rhs=0, symbols=[], exclude=[]):
    """
    Return a tuple derived from ``f = lhs - rhs`` that is one of
    the following: ``(0, 1)``, ``(0, 0)``, ``(symbol, solution)``, ``(n, d)``.

    Explanation
    ===========

    ``(0, 1)`` meaning that ``f`` is independent of the symbols in *symbols*
    that are not in *exclude*.

    ``(0, 0)`` meaning that there is no solution to the equation amongst the
    symbols given. If the first element of the tuple is not zero, then the
    function is guaranteed to be dependent on a symbol in *symbols*.

    ``(symbol, solution)`` where symbol appears linearly in the numerator of
    ``f``, is in *symbols* (if given), and is not in *exclude* (if given). No
    simplification is done to ``f`` other than a ``mul=True`` expansion, so the
    solution will correspond strictly to a unique solution.

    ``(n, d)`` where ``n`` and ``d`` are the numerator and denominator of ``f``
    when the numerator was not linear in any symbol of interest; ``n`` will
    never be a symbol unless a solution for that symbol was found (in which case
    the second element is the solution, not the denominator).

    Examples
    ========

    >>> from sympy import cancel, Pow

    ``f`` is independent of the symbols in *symbols* that are not in
    *exclude*:

    >>> from sympy import cos, sin, solve_linear
    >>> from sympy.abc import x, y, z
    >>> eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    >>> solve_linear(eq)
    (0, 1)
    >>> eq = cos(x)**2 + sin(x)**2  # = 1
    >>> solve_linear(eq)
    (0, 1)
    >>> solve_linear(x, exclude=[x])
    (0, 1)

    The variable ``x`` appears as a linear variable in each of the
    following:

    >>> solve_linear(x + y**2)
    (x, -y**2)
    >>> solve_linear(1/x - y**2)
    (x, y**(-2))

    When not linear in ``x`` or ``y`` then the numerator and denominator are
    returned:

    >>> solve_linear(x**2/y**2 - 3)
    (x**2 - 3*y**2, y**2)

    If the numerator of the expression is a symbol, then ``(0, 0)`` is
    returned if the solution for that symbol would have set any
    denominator to 0:

    >>> eq = 1/(1/x - 2)
    >>> eq.as_numer_denom()
    (x, 1 - 2*x)
    >>> solve_linear(eq)
    (0, 0)

    But automatic rewriting may cause a symbol in the denominator to
    appear in the numerator so a solution will be returned:

    >>> (1/x)**-1
    x
    >>> solve_linear((1/x)**-1)
    (x, 0)

    Use an unevaluated expression to avoid this:

    >>> solve_linear(Pow(1/x, -1, evaluate=False))
    (0, 0)

    If ``x`` is allowed to cancel in the following expression, then it
    appears to be linear in ``x``, but this sort of cancellation is not
    done by ``solve_linear`` so the solution will always satisfy the
    original expression without causing a division by zero error.

    >>> eq = x**2*(1/x - z**2/x)
    >>> solve_linear(cancel(eq))
    (x, 0)
    >>> solve_linear(eq)
    (x**2*(1 - z**2), x)

    A list of symbols for which a solution is desired may be given:

    >>> solve_linear(x + y + z, symbols=[y])
    (y, -x - z)

    A list of symbols to ignore may also be given:

    >>> solve_linear(x + y + z, exclude=[x])
    (y, -x - z)

    (A solution for ``y`` is obtained because it is the first variable
    from the canonically sorted list of symbols that had a linear
    solution.)

    """
    if isinstance(lhs, Eq):
        if rhs:
            raise ValueError(filldedent('\n            If lhs is an Equality, rhs must be 0 but was %s' % rhs))
        rhs = lhs.rhs
        lhs = lhs.lhs
    dens = None
    eq = lhs - rhs
    n, d = eq.as_numer_denom()
    if not n:
        return (S.Zero, S.One)
    free = n.free_symbols
    if not symbols:
        symbols = free
    else:
        bad = [s for s in symbols if not s.is_Symbol]
        if bad:
            if len(bad) == 1:
                bad = bad[0]
            if len(symbols) == 1:
                eg = 'solve(%s, %s)' % (eq, symbols[0])
            else:
                eg = 'solve(%s, *%s)' % (eq, list(symbols))
            raise ValueError(filldedent('\n                solve_linear only handles symbols, not %s. To isolate\n                non-symbols use solve, e.g. >>> %s <<<.\n                             ' % (bad, eg)))
        symbols = free.intersection(symbols)
    symbols = symbols.difference(exclude)
    if not symbols:
        return (S.Zero, S.One)
    derivs = defaultdict(list)
    for der in n.atoms(Derivative):
        csym = der.free_symbols & symbols
        for c in csym:
            derivs[c].append(der)
    all_zero = True
    for xi in sorted(symbols, key=default_sort_key):
        if isinstance(derivs[xi], list):
            derivs[xi] = {der: der.doit() for der in derivs[xi]}
        newn = n.subs(derivs[xi])
        dnewn_dxi = newn.diff(xi)
        free = dnewn_dxi.free_symbols
        if dnewn_dxi and (not free or any((dnewn_dxi.diff(s) for s in free)) or free == symbols):
            all_zero = False
            if dnewn_dxi is S.NaN:
                break
            if xi not in dnewn_dxi.free_symbols:
                vi = -1 / dnewn_dxi * newn.subs(xi, 0)
                if dens is None:
                    dens = _simple_dens(eq, symbols)
                if not any((checksol(di, {xi: vi}, minimal=True) is True for di in dens)):
                    irep = [(i, i.doit()) for i in vi.atoms(Integral) if i.function.is_number]
                    vi = expand_mul(vi.subs(irep))
                    return (xi, vi)
    if all_zero:
        return (S.Zero, S.One)
    if n.is_Symbol:
        return (S.Zero, S.Zero)
    return (n, d)