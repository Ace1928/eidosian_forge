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
def solve_univariate_inequality(expr, gen, relational=True, domain=S.Reals, continuous=False):
    """Solves a real univariate inequality.

    Parameters
    ==========

    expr : Relational
        The target inequality
    gen : Symbol
        The variable for which the inequality is solved
    relational : bool
        A Relational type output is expected or not
    domain : Set
        The domain over which the equation is solved
    continuous: bool
        True if expr is known to be continuous over the given domain
        (and so continuous_domain() does not need to be called on it)

    Raises
    ======

    NotImplementedError
        The solution of the inequality cannot be determined due to limitation
        in :func:`sympy.solvers.solveset.solvify`.

    Notes
    =====

    Currently, we cannot solve all the inequalities due to limitations in
    :func:`sympy.solvers.solveset.solvify`. Also, the solution returned for trigonometric inequalities
    are restricted in its periodic interval.

    See Also
    ========

    sympy.solvers.solveset.solvify: solver returning solveset solutions with solve's output API

    Examples
    ========

    >>> from sympy import solve_univariate_inequality, Symbol, sin, Interval, S
    >>> x = Symbol('x')

    >>> solve_univariate_inequality(x**2 >= 4, x)
    ((2 <= x) & (x < oo)) | ((-oo < x) & (x <= -2))

    >>> solve_univariate_inequality(x**2 >= 4, x, relational=False)
    Union(Interval(-oo, -2), Interval(2, oo))

    >>> domain = Interval(0, S.Infinity)
    >>> solve_univariate_inequality(x**2 >= 4, x, False, domain)
    Interval(2, oo)

    >>> solve_univariate_inequality(sin(x) > 0, x, relational=False)
    Interval.open(0, pi)

    """
    from sympy.solvers.solvers import denoms
    if domain.is_subset(S.Reals) is False:
        raise NotImplementedError(filldedent('\n        Inequalities in the complex domain are\n        not supported. Try the real domain by\n        setting domain=S.Reals'))
    elif domain is not S.Reals:
        rv = solve_univariate_inequality(expr, gen, relational=False, continuous=continuous).intersection(domain)
        if relational:
            rv = rv.as_relational(gen)
        return rv
    else:
        pass
    _gen = gen
    _domain = domain
    if gen.is_extended_real is False:
        rv = S.EmptySet
        return rv if not relational else rv.as_relational(_gen)
    elif gen.is_extended_real is None:
        gen = Dummy('gen', extended_real=True)
        try:
            expr = expr.xreplace({_gen: gen})
        except TypeError:
            raise TypeError(filldedent('\n                When gen is real, the relational has a complex part\n                which leads to an invalid comparison like I < 0.\n                '))
    rv = None
    if expr is S.true:
        rv = domain
    elif expr is S.false:
        rv = S.EmptySet
    else:
        e = expr.lhs - expr.rhs
        period = periodicity(e, gen)
        if period == S.Zero:
            e = expand_mul(e)
            const = expr.func(e, 0)
            if const is S.true:
                rv = domain
            elif const is S.false:
                rv = S.EmptySet
        elif period is not None:
            frange = function_range(e, gen, domain)
            rel = expr.rel_op
            if rel in ('<', '<='):
                if expr.func(frange.sup, 0):
                    rv = domain
                elif not expr.func(frange.inf, 0):
                    rv = S.EmptySet
            elif rel in ('>', '>='):
                if expr.func(frange.inf, 0):
                    rv = domain
                elif not expr.func(frange.sup, 0):
                    rv = S.EmptySet
            inf, sup = (domain.inf, domain.sup)
            if sup - inf is S.Infinity:
                domain = Interval(0, period, False, True).intersect(_domain)
                _domain = domain
        if rv is None:
            n, d = e.as_numer_denom()
            try:
                if gen not in n.free_symbols and len(e.free_symbols) > 1:
                    raise ValueError
                solns = solvify(e, gen, domain)
                if solns is None:
                    raise ValueError
            except (ValueError, NotImplementedError):
                raise NotImplementedError(filldedent('\n                    The inequality, %s, cannot be solved using\n                    solve_univariate_inequality.\n                    ' % expr.subs(gen, Symbol('x'))))
            expanded_e = expand_mul(e)

            def valid(x):
                v = expanded_e.subs(gen, expand_mul(x))
                try:
                    r = expr.func(v, 0)
                except TypeError:
                    r = S.false
                if r in (S.true, S.false):
                    return r
                if v.is_extended_real is False:
                    return S.false
                else:
                    v = v.n(2)
                    if v.is_comparable:
                        return expr.func(v, 0)
                    raise NotImplementedError('relationship did not evaluate: %s' % r)
            singularities = []
            for d in denoms(expr, gen):
                singularities.extend(solvify(d, gen, domain))
            if not continuous:
                domain = continuous_domain(expanded_e, gen, domain)
            include_x = '=' in expr.rel_op and expr.rel_op != '!='
            try:
                discontinuities = set(domain.boundary - FiniteSet(domain.inf, domain.sup))
                critical_points = FiniteSet(*solns + singularities + list(discontinuities)).intersection(Interval(domain.inf, domain.sup, domain.inf not in domain, domain.sup not in domain))
                if all((r.is_number for r in critical_points)):
                    reals = _nsort(critical_points, separated=True)[0]
                else:
                    sifted = sift(critical_points, lambda x: x.is_extended_real)
                    if sifted[None]:
                        raise NotImplementedError
                    try:
                        reals = sifted[True]
                        if len(reals) > 1:
                            reals = sorted(reals)
                    except TypeError:
                        raise NotImplementedError
            except NotImplementedError:
                raise NotImplementedError('sorting of these roots is not supported')
            make_real = S.Reals
            if im(expanded_e) != S.Zero:
                check = True
                im_sol = FiniteSet()
                try:
                    a = solveset(im(expanded_e), gen, domain)
                    if not isinstance(a, Interval):
                        for z in a:
                            if z not in singularities and valid(z) and z.is_extended_real:
                                im_sol += FiniteSet(z)
                    else:
                        start, end = (a.inf, a.sup)
                        for z in _nsort(critical_points + FiniteSet(end)):
                            valid_start = valid(start)
                            if start != end:
                                valid_z = valid(z)
                                pt = _pt(start, z)
                                if pt not in singularities and pt.is_extended_real and valid(pt):
                                    if valid_start and valid_z:
                                        im_sol += Interval(start, z)
                                    elif valid_start:
                                        im_sol += Interval.Ropen(start, z)
                                    elif valid_z:
                                        im_sol += Interval.Lopen(start, z)
                                    else:
                                        im_sol += Interval.open(start, z)
                            start = z
                        for s in singularities:
                            im_sol -= FiniteSet(s)
                except TypeError:
                    im_sol = S.Reals
                    check = False
                if im_sol is S.EmptySet:
                    raise ValueError(filldedent('\n                        %s contains imaginary parts which cannot be\n                        made 0 for any value of %s satisfying the\n                        inequality, leading to relations like I < 0.\n                        ' % (expr.subs(gen, _gen), _gen)))
                make_real = make_real.intersect(im_sol)
            sol_sets = [S.EmptySet]
            start = domain.inf
            if start in domain and valid(start) and start.is_finite:
                sol_sets.append(FiniteSet(start))
            for x in reals:
                end = x
                if valid(_pt(start, end)):
                    sol_sets.append(Interval(start, end, True, True))
                if x in singularities:
                    singularities.remove(x)
                else:
                    if x in discontinuities:
                        discontinuities.remove(x)
                        _valid = valid(x)
                    else:
                        _valid = include_x
                    if _valid:
                        sol_sets.append(FiniteSet(x))
                start = end
            end = domain.sup
            if end in domain and valid(end) and end.is_finite:
                sol_sets.append(FiniteSet(end))
            if valid(_pt(start, end)):
                sol_sets.append(Interval.open(start, end))
            if im(expanded_e) != S.Zero and check:
                rv = make_real.intersect(_domain)
            else:
                rv = Intersection(Union(*sol_sets), make_real, _domain).subs(gen, _gen)
    return rv if not relational else rv.as_relational(_gen)