import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def ineq(xl, xu, yl, yu):
    """Compute the "bounds" on an InequalityExpression

    Note this is *not* performing interval arithmetic: we are
    calculating the "bounds" on a RelationalExpression (whose domain is
    {True, False}).  Therefore we are determining if `x` can be less
    than `y`, `x` can not be less than `y`, or both.

    """
    ans = []
    if yl < xu:
        ans.append(_false)
    if xl <= yu:
        ans.append(_true)
    assert ans
    if len(ans) == 1:
        ans.append(ans[0])
    return tuple(ans)