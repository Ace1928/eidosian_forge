from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def polylog_general(ctx, s, z):
    v = ctx.zero
    u = ctx.ln(z)
    if not abs(u) < 5:
        j = ctx.j
        v = 1 - s
        y = ctx.ln(-z) / (2 * ctx.pi * j)
        return ctx.gamma(v) * (j ** v * ctx.zeta(v, 0.5 + y) + j ** (-v) * ctx.zeta(v, 0.5 - y)) / (2 * ctx.pi) ** v
    t = 1
    k = 0
    while 1:
        term = ctx.zeta(s - k) * t
        if abs(term) < ctx.eps:
            break
        v += term
        k += 1
        t *= u
        t /= k
    return ctx.gamma(1 - s) * (-u) ** (s - 1) + v