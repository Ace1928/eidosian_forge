from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def polylog_series(ctx, s, z):
    tol = +ctx.eps
    l = ctx.zero
    k = 1
    zk = z
    while 1:
        term = zk / k ** s
        l += term
        if abs(term) < tol:
            break
        zk *= z
        k += 1
    return l