from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def primepi2(ctx, x):
    x = int(x)
    if x < 2:
        return ctx._iv.zero
    if x < 2657:
        return ctx._iv.mpf(ctx.primepi(x))
    mid = ctx.li(x)
    err = ctx.sqrt(x, rounding='u') * ctx.ln(x, rounding='u') / 8 / ctx.pi(rounding='d')
    a = ctx.floor((ctx._iv.mpf(mid) - err).a, rounding='d')
    b = ctx.ceil((ctx._iv.mpf(mid) + err).b, rounding='u')
    return ctx._iv.mpf([a, b])