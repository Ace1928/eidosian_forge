from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
from .singularities import singularities
from sympy.core import Pow, S
from sympy.core.function import diff, expand_mul
from sympy.core.kind import NumberKind
from sympy.core.mod import Mod
from sympy.core.numbers import equal_valued
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.polys.polytools import degree, lcm_list
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
from sympy.sets.fancysets import ImageSet
from sympy.utilities import filldedent
from sympy.utilities.iterables import iterable
def function_range(f, symbol, domain):
    """
    Finds the range of a function in a given domain.
    This method is limited by the ability to determine the singularities and
    determine limits.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the range of function is to be determined.
    domain : :py:class:`~.Interval`
        The domain under which the range of the function has to be found.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, exp, log, pi, sqrt, sin, tan
    >>> from sympy.calculus.util import function_range
    >>> x = Symbol('x')
    >>> function_range(sin(x), x, Interval(0, 2*pi))
    Interval(-1, 1)
    >>> function_range(tan(x), x, Interval(-pi/2, pi/2))
    Interval(-oo, oo)
    >>> function_range(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> function_range(exp(x), x, S.Reals)
    Interval.open(0, oo)
    >>> function_range(log(x), x, S.Reals)
    Interval(-oo, oo)
    >>> function_range(sqrt(x), x, Interval(-5, 9))
    Interval(0, 3)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all ranges for all intervals under domain where function is
        continuous.

    Raises
    ======

    NotImplementedError
        If any of the intervals, in the given domain, for which function
        is continuous are not finite or real,
        OR if the critical points of the function on the domain cannot be found.
    """
    if domain is S.EmptySet:
        return S.EmptySet
    period = periodicity(f, symbol)
    if period == S.Zero:
        return FiniteSet(f.expand())
    from sympy.series.limits import limit
    from sympy.solvers.solveset import solveset
    if period is not None:
        if isinstance(domain, Interval):
            if (domain.inf - domain.sup).is_infinite:
                domain = Interval(0, period)
        elif isinstance(domain, Union):
            for sub_dom in domain.args:
                if isinstance(sub_dom, Interval) and (sub_dom.inf - sub_dom.sup).is_infinite:
                    domain = Interval(0, period)
    intervals = continuous_domain(f, symbol, domain)
    range_int = S.EmptySet
    if isinstance(intervals, (Interval, FiniteSet)):
        interval_iter = (intervals,)
    elif isinstance(intervals, Union):
        interval_iter = intervals.args
    else:
        raise NotImplementedError(filldedent('\n                Unable to find range for the given domain.\n                '))
    for interval in interval_iter:
        if isinstance(interval, FiniteSet):
            for singleton in interval:
                if singleton in domain:
                    range_int += FiniteSet(f.subs(symbol, singleton))
        elif isinstance(interval, Interval):
            vals = S.EmptySet
            critical_points = S.EmptySet
            critical_values = S.EmptySet
            bounds = ((interval.left_open, interval.inf, '+'), (interval.right_open, interval.sup, '-'))
            for is_open, limit_point, direction in bounds:
                if is_open:
                    critical_values += FiniteSet(limit(f, symbol, limit_point, direction))
                    vals += critical_values
                else:
                    vals += FiniteSet(f.subs(symbol, limit_point))
            solution = solveset(f.diff(symbol), symbol, interval)
            if not iterable(solution):
                raise NotImplementedError('Unable to find critical points for {}'.format(f))
            if isinstance(solution, ImageSet):
                raise NotImplementedError('Infinite number of critical points for {}'.format(f))
            critical_points += solution
            for critical_point in critical_points:
                vals += FiniteSet(f.subs(symbol, critical_point))
            left_open, right_open = (False, False)
            if critical_values is not S.EmptySet:
                if critical_values.inf == vals.inf:
                    left_open = True
                if critical_values.sup == vals.sup:
                    right_open = True
            range_int += Interval(vals.inf, vals.sup, left_open, right_open)
        else:
            raise NotImplementedError(filldedent('\n                Unable to find range for the given domain.\n                '))
    return range_int