import itertools
from sympy.calculus.util import (continuous_domain, periodicity,
from sympy.core import Symbol, Dummy, sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Eq, Ge, Lt
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import im, Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent
def solve_rational_inequalities(eqs):
    """Solve a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import solve_rational_inequalities, Poly

    >>> solve_rational_inequalities([[
    ... ((Poly(-x + 1), Poly(1, x)), '>='),
    ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
    {1}

    >>> solve_rational_inequalities([[
    ... ((Poly(x), Poly(1, x)), '!='),
    ... ((Poly(-x + 1), Poly(1, x)), '>=')]])
    Union(Interval.open(-oo, 0), Interval.Lopen(0, 1))

    See Also
    ========
    solve_poly_inequality
    """
    result = S.EmptySet
    for _eqs in eqs:
        if not _eqs:
            continue
        global_intervals = [Interval(S.NegativeInfinity, S.Infinity)]
        for (numer, denom), rel in _eqs:
            numer_intervals = solve_poly_inequality(numer * denom, rel)
            denom_intervals = solve_poly_inequality(denom, '==')
            intervals = []
            for numer_interval, global_interval in itertools.product(numer_intervals, global_intervals):
                interval = numer_interval.intersect(global_interval)
                if interval is not S.EmptySet:
                    intervals.append(interval)
            global_intervals = intervals
            intervals = []
            for global_interval in global_intervals:
                for denom_interval in denom_intervals:
                    global_interval -= denom_interval
                if global_interval is not S.EmptySet:
                    intervals.append(global_interval)
            global_intervals = intervals
            if not global_intervals:
                break
        for interval in global_intervals:
            result = result.union(interval)
    return result