from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def grampoint(ctx, n):
    g = 2 * ctx.pi * ctx.exp(1 + ctx.lambertw((8 * n + 1) / (8 * ctx.e)))
    return ctx.findroot(lambda t: ctx.siegeltheta(t) - ctx.pi * n, g)