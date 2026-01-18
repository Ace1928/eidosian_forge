from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def clcos(ctx, s, z, pi=False):
    if ctx.isint(s) and s < 0 and (int(s) % 2 == 0):
        return z * 0
    if pi:
        a = ctx.expjpi(z)
    else:
        a = ctx.expj(z)
    if ctx._is_real_type(z) and ctx._is_real_type(s):
        return ctx.re(ctx.polylog(s, a))
    b = 1 / a
    return 0.5 * (ctx.polylog(s, a) + ctx.polylog(s, b))