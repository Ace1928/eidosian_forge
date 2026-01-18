from copy import copy
from ..libmp.backend import xrange
def improve_solution(ctx, A, x, b, maxsteps=1):
    """
        Improve a solution to a linear equation system iteratively.

        This re-uses the LU decomposition and is thus cheap.
        Usually 3 up to 4 iterations are giving the maximal improvement.
        """
    if A.rows != A.cols:
        raise RuntimeError('need n*n matrix')
    for _ in xrange(maxsteps):
        r = ctx.residual(A, x, b)
        if ctx.norm(r, 2) < 10 * ctx.eps:
            break
        dx = ctx.lu_solve(A, -r)
        x += dx
    return x