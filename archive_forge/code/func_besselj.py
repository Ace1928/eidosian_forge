from .functions import defun, defun_wrapped
@defun
def besselj(ctx, n, z, derivative=0, **kwargs):
    if type(n) is int:
        n_isint = True
    else:
        n = ctx.convert(n)
        n_isint = ctx.isint(n)
        if n_isint:
            n = int(ctx._re(n))
    if n_isint and n < 0:
        return (-1) ** n * ctx.besselj(-n, z, derivative, **kwargs)
    z = ctx.convert(z)
    M = ctx.mag(z)
    if derivative:
        d = ctx.convert(derivative)
        if ctx.isint(d) and d >= 0:
            d = int(d)
            orig = ctx.prec
            try:
                ctx.prec += 15
                v = ctx.fsum(((-1) ** k * ctx.binomial(d, k) * ctx.besselj(2 * k + n - d, z) for k in range(d + 1)))
            finally:
                ctx.prec = orig
            v *= ctx.mpf(2) ** (-d)
        else:

            def h(n, d):
                r = ctx.fmul(ctx.fmul(z, z, prec=ctx.prec + M), -0.25, exact=True)
                B = [0.5 * (n - d + 1), 0.5 * (n - d + 2)]
                T = [([2, ctx.pi, z], [d - 2 * n, 0.5, n - d], [], B, [(n + 1) * 0.5, (n + 2) * 0.5], B + [n + 1], r)]
                return T
            v = ctx.hypercomb(h, [n, d], **kwargs)
    else:
        if not derivative and n_isint and (abs(M) < 10) and (abs(n) < 20):
            try:
                return ctx._besselj(n, z)
            except NotImplementedError:
                pass
        if not z:
            if not n:
                v = ctx.one + n + z
            elif ctx.re(n) > 0:
                v = n * z
            else:
                v = ctx.inf + z + n
        else:
            orig = ctx.prec
            try:
                ctx.prec += min(3 * abs(M), ctx.prec)
                w = ctx.fmul(z, 0.5, exact=True)

                def h(n):
                    r = ctx.fneg(ctx.fmul(w, w, prec=max(0, ctx.prec + M)), exact=True)
                    return [([w], [n], [], [n + 1], [], [n + 1], r)]
                v = ctx.hypercomb(h, [n], **kwargs)
            finally:
                ctx.prec = orig
        v = +v
    return v