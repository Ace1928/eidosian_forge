import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def ranged(xl, xu, yl, yu, zl, zu):
    """Compute the "bounds" on a RangedExpression

    Note this is *not* performing interval arithmetic: we are
    calculating the "bounds" on a RelationalExpression (whose domain is
    {True, False}).  Therefore we are determining if `y` can be between
    `z` and `z`, `y` can be outside the range `x` and `z`, or both.

    """
    lb = ineq(xl, xu, yl, yu)
    ub = ineq(yl, yu, zl, zu)
    ans = []
    if not lb[0] or not ub[0]:
        ans.append(_false)
    if lb[1] and ub[1]:
        ans.append(_true)
    if len(ans) == 1:
        ans.append(ans[0])
    return tuple(ans)