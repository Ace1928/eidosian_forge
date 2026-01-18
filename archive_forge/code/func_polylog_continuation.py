from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def polylog_continuation(ctx, n, z):
    if n < 0:
        return z * 0
    twopij = 2j * ctx.pi
    a = -twopij ** n / ctx.fac(n) * ctx.bernpoly(n, ctx.ln(z) / twopij)
    if ctx._is_real_type(z) and z < 0:
        a = ctx._re(a)
    if ctx._im(z) < 0 or (ctx._im(z) == 0 and ctx._re(z) >= 1):
        a -= twopij * ctx.ln(z) ** (n - 1) / ctx.fac(n - 1)
    return a