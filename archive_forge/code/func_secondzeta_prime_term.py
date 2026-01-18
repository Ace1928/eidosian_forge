from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def secondzeta_prime_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5 * (1 - s), 0.25 * ctx.log(n) ** 2 * a ** (-1)) * (0.5 * ctx.log(n)) ** (s - 1) * ctx.mangoldt(n) / ctx.sqrt(n) / (2 * ctx.gamma(0.5 * s) * ctx.sqrt(ctx.pi))
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 1
    while mg > tol or n < 9:
        totsum += term
        n += 1
        term = f(n)
        if term == 0:
            mg = ctx.inf
        else:
            mg = abs(term)
    if kwargs.get('error'):
        err = mg
    return (+totsum, err, n)