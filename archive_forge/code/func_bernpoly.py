from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
@defun_wrapped
def bernpoly(ctx, n, z):
    n = int(n)
    if n < 0:
        raise ValueError('Bernoulli polynomials only defined for n >= 0')
    if z == 0 or (z == 1 and n > 1):
        return ctx.bernoulli(n)
    if z == 0.5:
        return (ctx.ldexp(1, 1 - n) - 1) * ctx.bernoulli(n)
    if n <= 3:
        if n == 0:
            return z ** 0
        if n == 1:
            return z - 0.5
        if n == 2:
            return (6 * z * (z - 1) + 1) / 6
        if n == 3:
            return z * (z * (z - 1.5) + 0.5)
    if ctx.isinf(z):
        return z ** n
    if ctx.isnan(z):
        return z
    if abs(z) > 2:

        def terms():
            t = ctx.one
            yield t
            r = ctx.one / z
            k = 1
            while k <= n:
                t = t * (n + 1 - k) / k * r
                if not (k > 2 and k & 1):
                    yield (t * ctx.bernoulli(k))
                k += 1
        return ctx.sum_accurately(terms) * z ** n
    else:

        def terms():
            yield ctx.bernoulli(n)
            t = ctx.one
            k = 1
            while k <= n:
                t = t * (n + 1 - k) / k * z
                m = n - k
                if not (m > 2 and m & 1):
                    yield (t * ctx.bernoulli(m))
                k += 1
        return ctx.sum_accurately(terms)