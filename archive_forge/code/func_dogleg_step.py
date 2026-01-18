import numpy as np
from numpy.linalg import lstsq, norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, lsmr
from scipy.optimize import OptimizeResult
from .common import (
def dogleg_step(x, newton_step, g, a, b, tr_bounds, lb, ub):
    """Find dogleg step in a rectangular region.

    Returns
    -------
    step : ndarray, shape (n,)
        Computed dogleg step.
    bound_hits : ndarray of int, shape (n,)
        Each component shows whether a corresponding variable hits the
        initial bound after the step is taken:
            *  0 - a variable doesn't hit the bound.
            * -1 - lower bound is hit.
            *  1 - upper bound is hit.
    tr_hit : bool
        Whether the step hit the boundary of the trust-region.
    """
    lb_total, ub_total, orig_l, orig_u, tr_l, tr_u = find_intersection(x, tr_bounds, lb, ub)
    bound_hits = np.zeros_like(x, dtype=int)
    if in_bounds(newton_step, lb_total, ub_total):
        return (newton_step, bound_hits, False)
    to_bounds, _ = step_size_to_bound(np.zeros_like(x), -g, lb_total, ub_total)
    cauchy_step = -minimize_quadratic_1d(a, b, 0, to_bounds)[0] * g
    step_diff = newton_step - cauchy_step
    step_size, hits = step_size_to_bound(cauchy_step, step_diff, lb_total, ub_total)
    bound_hits[(hits < 0) & orig_l] = -1
    bound_hits[(hits > 0) & orig_u] = 1
    tr_hit = np.any((hits < 0) & tr_l | (hits > 0) & tr_u)
    return (cauchy_step + step_size * step_diff, bound_hits, tr_hit)