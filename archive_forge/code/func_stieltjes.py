from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun
def stieltjes(ctx, n, a=1):
    n = ctx.convert(n)
    a = ctx.convert(a)
    if n < 0:
        return ctx.bad_domain('Stieltjes constants defined for n >= 0')
    if hasattr(ctx, 'stieltjes_cache'):
        stieltjes_cache = ctx.stieltjes_cache
    else:
        stieltjes_cache = ctx.stieltjes_cache = {}
    if a == 1:
        if n == 0:
            return +ctx.euler
        if n in stieltjes_cache:
            prec, s = stieltjes_cache[n]
            if prec >= ctx.prec:
                return +s
    mag = 1

    def f(x):
        xa = x / a
        v = (xa - ctx.j) * ctx.ln(a - ctx.j * x) ** n / (1 + xa ** 2) / (ctx.exp(2 * ctx.pi * x) - 1)
        return ctx._re(v) / mag
    orig = ctx.prec
    try:
        if n > 50:
            ctx.prec = 20
            mag = ctx.quad(f, [0, ctx.inf], maxdegree=3)
        ctx.prec = orig + 10 + int(n ** 0.5)
        s = ctx.quad(f, [0, ctx.inf], maxdegree=20)
        v = ctx.ln(a) ** n / (2 * a) - ctx.ln(a) ** (n + 1) / (n + 1) + 2 * s / a * mag
    finally:
        ctx.prec = orig
    if a == 1 and ctx.isint(n):
        stieltjes_cache[n] = (ctx.prec, v)
    return +v