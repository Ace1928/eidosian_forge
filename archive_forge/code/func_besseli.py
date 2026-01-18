from .functions import defun, defun_wrapped
@defun
def besseli(ctx, n, z, derivative=0, **kwargs):
    n = ctx.convert(n)
    z = ctx.convert(z)
    if not z:
        if derivative:
            raise ValueError
        if not n:
            return 1 + n + z
        if ctx.isint(n):
            return 0 * (n + z)
        r = ctx.re(n)
        if r == 0:
            return ctx.nan * (n + z)
        elif r > 0:
            return 0 * (n + z)
        else:
            return ctx.inf + (n + z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)

        def h(n, d):
            r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec + M), 0.25, exact=True)
            B = [0.5 * (n - d + 1), 0.5 * (n - d + 2), n + 1]
            T = [([2, ctx.pi, z], [d - 2 * n, 0.5, n - d], [n + 1], B, [(n + 1) * 0.5, (n + 2) * 0.5], B, r)]
            return T
        v = ctx.hypercomb(h, [n, d], **kwargs)
    else:

        def h(n):
            w = ctx.fmul(z, 0.5, exact=True)
            r = ctx.fmul(w, w, prec=max(0, ctx.prec + M))
            return [([w], [n], [], [n + 1], [], [n + 1], r)]
        v = ctx.hypercomb(h, [n], **kwargs)
    return v