from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def as_expr_set_pairs(self, domain=None):
    """Return tuples for each argument of self that give
        the expression and the interval in which it is valid
        which is contained within the given domain.
        If a condition cannot be converted to a set, an error
        will be raised. The variable of the conditions is
        assumed to be real; sets of real values are returned.

        Examples
        ========

        >>> from sympy import Piecewise, Interval
        >>> from sympy.abc import x
        >>> p = Piecewise(
        ...     (1, x < 2),
        ...     (2,(x > 0) & (x < 4)),
        ...     (3, True))
        >>> p.as_expr_set_pairs()
        [(1, Interval.open(-oo, 2)),
         (2, Interval.Ropen(2, 4)),
         (3, Interval(4, oo))]
        >>> p.as_expr_set_pairs(Interval(0, 3))
        [(1, Interval.Ropen(0, 2)),
         (2, Interval(2, 3))]
        """
    if domain is None:
        domain = S.Reals
    exp_sets = []
    U = domain
    complex = not domain.is_subset(S.Reals)
    cond_free = set()
    for expr, cond in self.args:
        cond_free |= cond.free_symbols
        if len(cond_free) > 1:
            raise NotImplementedError(filldedent('\n                    multivariate conditions are not handled.'))
        if complex:
            for i in cond.atoms(Relational):
                if not isinstance(i, (Eq, Ne)):
                    raise ValueError(filldedent('\n                            Inequalities in the complex domain are\n                            not supported. Try the real domain by\n                            setting domain=S.Reals'))
        cond_int = U.intersect(cond.as_set())
        U = U - cond_int
        if cond_int != S.EmptySet:
            exp_sets.append((expr, cond_int))
    return exp_sets