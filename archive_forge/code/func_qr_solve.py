from copy import copy
from ..libmp.backend import xrange
def qr_solve(ctx, A, b, norm=None, **kwargs):
    """
        Ax = b => x, ||Ax - b||

        Solve a determined or overdetermined linear equations system and
        calculate the norm of the residual (error).
        QR decomposition using Householder factorization is applied, which gives very
        accurate results even for ill-conditioned matrices. qr_solve is twice as
        efficient.
        """
    if norm is None:
        norm = ctx.norm
    prec = ctx.prec
    try:
        ctx.prec += 10
        A, b = (ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy())
        if A.rows < A.cols:
            raise ValueError('cannot solve underdetermined system')
        H, p, x, r = ctx.householder(ctx.extend(A, b))
        res = ctx.norm(r)
        if res == 0:
            res = ctx.norm(ctx.residual(A, x, b))
        return (ctx.matrix(x, **kwargs), res)
    finally:
        ctx.prec = prec