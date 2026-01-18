from ..libmp.backend import xrange
from .functions import defun, defun_wrapped
@defun
def meijerg(ctx, a_s, b_s, z, r=1, series=None, **kwargs):
    an, ap = a_s
    bm, bq = b_s
    n = len(an)
    p = n + len(ap)
    m = len(bm)
    q = m + len(bq)
    a = an + ap
    b = bm + bq
    a = [ctx.convert(_) for _ in a]
    b = [ctx.convert(_) for _ in b]
    z = ctx.convert(z)
    if series is None:
        if p < q:
            series = 1
        if p > q:
            series = 2
        if p == q:
            if m + n == p and abs(z) > 1:
                series = 2
            else:
                series = 1
    if kwargs.get('verbose'):
        print('Meijer G m,n,p,q,series =', m, n, p, q, series)
    if series == 1:

        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(m):
                bases = [z]
                expts = [b[k] / r]
                gn = [b[j] - b[k] for j in range(m) if j != k]
                gn += [1 - a[j] + b[k] for j in range(n)]
                gd = [a[j] - b[k] for j in range(n, p)]
                gd += [1 - b[j] + b[k] for j in range(m, q)]
                hn = [1 - a[j] + b[k] for j in range(p)]
                hd = [1 - b[j] + b[k] for j in range(q) if j != k]
                hz = (-ctx.one) ** (p - m - n) * z ** (ctx.one / r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    else:

        def h(*args):
            a = args[:p]
            b = args[p:]
            terms = []
            for k in range(n):
                bases = [z]
                if r == 1:
                    expts = [a[k] - 1]
                else:
                    expts = [(a[k] - 1) / ctx.convert(r)]
                gn = [a[k] - a[j] for j in range(n) if j != k]
                gn += [1 - a[k] + b[j] for j in range(m)]
                gd = [a[k] - b[j] for j in range(m, q)]
                gd += [1 - a[k] + a[j] for j in range(n, p)]
                hn = [1 - a[k] + b[j] for j in range(q)]
                hd = [1 + a[j] - a[k] for j in range(p) if j != k]
                hz = (-ctx.one) ** (q - m - n) / z ** (ctx.one / r)
                terms.append((bases, expts, gn, gd, hn, hd, hz))
            return terms
    return ctx.hypercomb(h, a + b, **kwargs)