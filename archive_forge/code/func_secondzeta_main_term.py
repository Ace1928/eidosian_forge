from __future__ import print_function
from ..libmp.backend import xrange
from .functions import defun, defun_wrapped, defun_static
def secondzeta_main_term(ctx, s, a, **kwargs):
    tol = ctx.eps
    f = lambda n: ctx.gammainc(0.5 * s, a * gamm ** 2, regularized=True) * gamm ** (-s)
    totsum = term = ctx.zero
    mg = ctx.inf
    n = 0
    while mg > tol:
        totsum += term
        n += 1
        gamm = ctx.im(ctx.zetazero_memoized(n))
        term = f(n)
        mg = abs(term)
    err = 0
    if kwargs.get('error'):
        sg = ctx.re(s)
        err = 0.5 * ctx.pi ** (-1) * max(1, sg) * a ** (sg - 0.5) * ctx.log(gamm / (2 * ctx.pi)) * ctx.gammainc(-0.5, a * gamm ** 2) / abs(ctx.gamma(s / 2))
        err = abs(err)
    return (+totsum, err, n)